use crate::errors::*;
use crate::fvec::FVec;
use crate::gbm::grad_booster::GradBooster;
use crate::model_reader::ModelReader;

struct ModelParam {
    /// number of features
    num_feature: usize,
    /// how many output group a single instance can produce
    /// this affects the behavior of number of output we have:
    /// suppose we have n instance and k group, output will be k*n
    num_output_group: usize,
}

impl ModelParam {
    fn read_from<T: ModelReader>(reader: &mut T) -> Result<ModelParam> {
        let (num_feature, num_output_group) = (
            reader.read_i32_le()? as usize,
            reader.read_i32_le()? as usize,
        );
        let mut reserved = [0i32; 32];
        reader.read_to_i32_buffer(&mut reserved)?;
        // read padding
        reader.read_i32_le()?;
        Ok(ModelParam {
            num_feature,
            num_output_group,
        })
    }
}

pub struct GBLinear {
    mparam: ModelParam,
    weights: Vec<f32>,
}

impl GBLinear {
    pub fn read_from<T: ModelReader>(with_pbuffer: bool, reader: &mut T) -> Result<Self> {
        let mparam = ModelParam::read_from(reader)?;
        // read padding
        reader.read_i32_le()?;
        let weights = reader.read_float_vec((mparam.num_feature + 1) * mparam.num_output_group)?;

        Ok(GBLinear { mparam, weights })
    }

    fn bias(&self, gid: usize) -> f32 {
        self.weight(self.mparam.num_feature, gid)
    }

    fn weight(&self, fid: usize, gid: usize) -> f32 {
        self.weights[(fid * self.mparam.num_output_group) + gid]
    }

    fn pred<F: FVec>(&self, feat: &F, gid: usize) -> f32 {
        let mut psum = self.bias(gid) as f32;
        for fid in 0..self.mparam.num_feature {
            match feat.fvalue(fid) {
                None => {}
                Some(feat_val) => {
                    psum += feat_val * self.weight(fid, gid);
                }
            }
        }
        psum
    }
}

impl<F: FVec> GradBooster<F> for GBLinear {
    fn predict(&self, feat: &F, ntree_limit: usize) -> Vec<f32> {
        (0..self.mparam.num_output_group)
            .map(|gid| self.pred(feat, gid))
            .collect()
    }

    fn predict_single(&self, feat: &F, ntree_limit: usize) -> f32 {
        if self.mparam.num_output_group != 1 {
            panic!("Can't invoke predict_single() because this model outputs multiple values");
        }
        self.pred(feat, 0)
    }

    fn predict_leaf(&self, feat: &F, ntree_limit: usize) -> Vec<usize> {
        unimplemented!("gblinear does not support predict leaf index")
    }
}
