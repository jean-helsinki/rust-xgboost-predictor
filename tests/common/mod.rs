pub mod loaders;
pub mod tasks;

pub mod types {
    use std::collections::{HashMap, LinkedList};
    use xgboost_predictor::fvec::FVecMap;
    use xgboost_predictor::predictor::Predictor;

    pub type TestMap = FVecMap<f32>;
    pub type DataItem = (usize, TestMap);
    pub type TestPredictor = Predictor<TestMap>;
}
