use crate::errors::*;
use crate::fvec::FVec;
use crate::gbm::gblinear::GBLinear;
use crate::gbm::gbtree::GBTree;
use crate::model_reader::ModelReader;

/// Interface of gradient boosting model
pub trait GradBooster<F: FVec> {
    /// Generates predictions for given feature vector
    fn predict(&self, feat: &F, ntree_limit: usize) -> Vec<f32>;
    /// Generates a prediction for given feature vector
    fn predict_single(&self, feat: &F, ntree_limit: usize) -> f32;
    /// Predicts the leaf index of each tree. This is only valid in gbtree predictor
    fn predict_leaf(&self, feat: &F, ntree_limit: usize) -> Vec<usize>;
}

pub fn load_grad_booster<F: FVec, T: ModelReader>(
    reader: &mut T,
    name_gbm: Vec<u8>,
    with_pbuffer: bool,
) -> Result<Box<dyn GradBooster<F> + Send>> {
    match name_gbm.as_slice() {
        b"gbtree" => Ok(Box::new(GBTree::read_from(with_pbuffer, reader, false)?)),
        b"gblinear" => Ok(Box::new(GBLinear::read_from(with_pbuffer, reader)?)),
        b"dart" => Ok(Box::new(GBTree::read_from(with_pbuffer, reader, true)?)),
        _ => Err(Error::from_kind(ErrorKind::UnsupportedModelType(
            String::from_utf8(name_gbm)?,
        ))),
    }
}
