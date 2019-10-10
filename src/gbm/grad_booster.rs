use crate::fvec::FVec;

/// Interface of gradient boosting model
pub trait GradBooster<F: FVec> {
    /// Generates predictions for given feature vector
    fn predict(feat: &F, ntree_limit: i32) -> Vec<f64>;
    /// Generates a prediction for given feature vector
    fn predict_single(feat: &F, ntree_limit: i32) -> f64;
    /// Predicts the leaf index of each tree. This is only valid in gbtree predictor
    fn predict_leaf(feat: &F, ntree_limit: i32) -> Vec<i32>;
}
