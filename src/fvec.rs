use std::collections::HashMap;
use std::f32;

pub trait ToFloat {
    #[inline]
    fn to_double(&self) -> f32;
}

impl ToFloat for f32 {
    #[inline]
    fn to_double(&self) -> f32 {
        return self.clone();
    }
}

impl ToFloat for f64 {
    #[inline]
    fn to_double(&self) -> f32 {
        return *self as f32;
    }
}

/// Interface of feature vector
pub trait FVec {
    /// get value for index
    fn fvalue(&self, index: usize) -> Option<f32>;
}

pub type FVecMap<T: ToFloat> = HashMap<usize, T>;

/// Feature vector based on vec
pub struct FVecArray<T: ToFloat> {
    values: Vec<T>,
    treats_zero_as_none: bool,
}

impl<T: ToFloat> FVec for FVecMap<T> {
    fn fvalue(&self, index: usize) -> Option<f32> {
        return Some(self.get(&index)?.to_double());
    }
}

impl<T: ToFloat> FVec for FVecArray<T> {
    fn fvalue(&self, index: usize) -> Option<f32> {
        if self.values.len() <= index {
            return None;
        } else {
            let result = self.values[index].to_double();
            if self.treats_zero_as_none && result == 0f32 {
                return None;
            } else {
                return Some(result);
            }
        }
    }
}
