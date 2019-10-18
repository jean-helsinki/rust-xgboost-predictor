use crate::fvec::FVec;
use crate::model_reader::{ModelReader};
use crate::gbm::gbtree::GBTree;
use crate::gbm::gblinear::GBLinear;
use crate::errors::*;

/// Interface of gradient boosting model
pub trait GradBooster<F: FVec> {
    /// Generates predictions for given feature vector
    fn predict(&self, feat: &F, ntree_limit: usize) -> Vec<f64>;
    /// Generates a prediction for given feature vector
    fn predict_single(&self, feat: &F, ntree_limit: usize) -> f64;
    /// Predicts the leaf index of each tree. This is only valid in gbtree predictor
    fn predict_leaf(&self, feat: &F, ntree_limit: usize) -> Vec<usize>;
}


pub fn load_grad_booster<F: FVec, T: ModelReader>(reader: &mut T, name_gbm: &[u8], with_pbuffer: bool) -> Result<Box<GradBooster<F>>>{
    match name_gbm {
        b"gbtree" => Ok(Box::new(GBTree::new(with_pbuffer, reader, false)?)),
        b"gblinear" => Ok(Box::new(GBLinear::new(with_pbuffer, reader)?)),
        b"dart" => Ok(Box::new(GBTree::new(with_pbuffer, reader, true)?)),
        _ => std::io::Error::new(),
    }
}
