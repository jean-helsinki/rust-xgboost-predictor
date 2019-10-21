use crate::model_reader::ModelReader;
use crate::functions::{ObjFunction, FunctionType, get_classify_func_type, get_classify_function};
use crate::gbm::grad_booster::GradBooster;
use crate::fvec::FVec;
use crate::errors::*;
use byteorder::{LE, ByteOrder};


struct ModelParam {
    /// global bias
    base_score: f32,
    /// number of features
    num_feature: usize,
    /// number of class, if it is multi-class classification
    num_class: i32,
    /// whether the model itself is saved with pbuffer
    saved_with_pbuffer: i32,
    /// reserved field
    reserved: [i32; 30],
}

impl ModelParam {
    fn new<T: ModelReader>(base_score: f32, num_feature: usize, reader: &mut T) -> Result<ModelParam> {
        let (num_class, saved_with_pbuffer) = (reader.read_i32_le()?, reader.read_i32_le()?);
        let mut reserved = [0i32; 30];
        reader.read_to_i32_buffer(&mut reserved)?;
        return Ok(ModelParam{
            base_score,
            num_feature,
            num_class,
            saved_with_pbuffer,
            reserved,
        })
    }
}

/// Predicts using the Xgboost model
pub struct Predictor<F: FVec> {
    mparam: ModelParam,
    //SparkModelParam sparkModelParam;
    obj_func: ObjFunction,
    gbm: Box<GradBooster<F>>,
}

impl<F: FVec> Predictor<F> {
    fn read_model_params<T: ModelReader>(reader: &mut T) -> Result<ModelParam> {
        let mut first4bytes = [0u8; 4];
        let mut next4bytes = [0u8; 4];
        reader.read_exact(&mut first4bytes)?;
        reader.read_exact(&mut next4bytes)?;

        let (base_score, num_feature) = if &first4bytes == b"binf" {
            (LE::read_f32(&next4bytes), reader.read_i32_le()? as usize)
        } else if &first4bytes[..3] == [0x00u8, 0x05, 0x5f] {
            // Model generated by xgboost4j-spark?
            unimplemented!("not implemented for xgboost4j-spark models")
        } else {
            (LE::read_f32(&first4bytes), LE::read_i32(&next4bytes) as usize)
        };

        return ModelParam::new(base_score, num_feature, reader);
    }

    /// Instantiates with the Xgboost model
    pub fn new<T: ModelReader>(reader: &mut T) -> Result<Predictor<F>> {
        let mparam = Predictor::<F>::read_model_params(reader)?;

        let name_obj = reader.read_u8_vec_len()?;
        let name_gbm = reader.read_u8_vec_len()?;

        let obj_func_type = get_classify_func_type(name_obj)?;
        let obj_func = get_classify_function(obj_func_type);
        let gbm =
            crate::gbm::grad_booster::load_grad_booster( reader, name_gbm, mparam.saved_with_pbuffer != 0)?;

        return Ok(Predictor{
            mparam,
            obj_func,
            gbm,
        });
    }

    fn predict_raw(&self, feat: &F, ntree_limit: usize) -> Vec<f64> {
        let mut preds = self.gbm.predict(feat, ntree_limit);
        for i in 0..preds.len() {
            preds[i] += self.mparam.base_score as f64;
        };
        preds
    }

    fn predict_single_raw(&self, feat: &F, ntree_limit: usize) -> f64 {
        return self.gbm.predict_single(feat, ntree_limit) + self.mparam.base_score as f64;
    }

    /// Generates predictions for given feature vector
    pub fn predict(&self, feat: &F, output_margin: bool, ntree_limit: usize) -> Vec<f64>{
        let mut preds = self.predict_raw(feat, ntree_limit);

        return if output_margin {
            (self.obj_func.vector)(&preds)
        } else {
            preds
        }
    }

    /// Generates a prediction for given feature vector
    pub fn predict_single(&self, feat: &F, output_margin: bool, ntree_limit: usize) -> f64 {
        let pred = self.predict_single_raw(feat, ntree_limit);
        return if output_margin {
            (self.obj_func.scalar)(pred)
        } else {
            pred
        }
    }
}
