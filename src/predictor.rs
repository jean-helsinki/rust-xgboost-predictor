use crate::model_reader::{ModelReader, ModelReadResult};
use crate::functions::ObjFunction;

struct ModelParam {
    /// global bias
    base_score: f32,
    /// number of features
    num_feature: u32,
    /// number of class, if it is multi-class classification
    num_class: i32,
    /// whether the model itself is saved with pbuffer
    saved_with_pbuffer: i32,
    /// reserved field
    reserved: [i32; 30],
}

impl ModelParam {
    fn new<T: ModelReader>(base_score: f32, num_feature: u32, reader: &mut T) -> ModelReadResult<ModelParam> {
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

struct Predictor {
    mparam: ModelParam,
    //SparkModelParam sparkModelParam;
    name_obj: String,
    name_gbm: String,
    obj: ObjFunction,
    //gbm: GradBooster,
}

impl Predictor {
    /*fn read_model_params<T: ModelReader>(reader: &mut T) -> ModelReadResult<()> {
        Ok(())
    }*/

    /*fn new() -> Predictor {
        Predictor{}
    }*/
}
