use crate::fvec::FVec;

/// Interface of gradient boosting model
pub trait GradBooster<F: FVec> {
    /// Generates predictions for given feature vector
    fn predict(&self, feat: &F, ntree_limit: usize) -> Vec<f64>;
    /// Generates a prediction for given feature vector
    fn predict_single(&self, feat: &F, ntree_limit: usize) -> f64;
    /// Predicts the leaf index of each tree. This is only valid in gbtree predictor
    fn predict_leaf(&self, feat: &F, ntree_limit: usize) -> Vec<usize>;
}
