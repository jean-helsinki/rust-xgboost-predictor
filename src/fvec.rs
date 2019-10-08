use std::collections::HashMap;
use std::f64;

pub trait ToDouble {
    #[inline]
    fn to_double(&self) -> f64;
}

impl ToDouble for f64 {
    #[inline]
    fn to_double(&self) -> f64 {
        return self.clone();
    }
}

impl ToDouble for f32 {
    #[inline]
    fn to_double(&self) -> f64 {
        return *self as f64;
    }
}

/// Interface of feature vector
pub trait FVec {
    /// get value for index
    fn fvalue(&self, index: usize) -> Option<f64>;
}

pub type FVecMap<T: ToDouble> = HashMap<usize, T>;

/// Feature vector based on vec
pub struct FVecArray<T: ToDouble> {
    values: Vec<T>,
    treats_zero_as_none: bool,
}

impl<T: ToDouble> FVec for FVecMap<T> {
    fn fvalue(&self, index: usize) -> Option<f64> {
        return Some(self.get(&index)?.to_double());
    }
}

impl<T: ToDouble> FVec for FVecArray<T> {
    fn fvalue(&self, index: usize) -> Option<f64> {
        if self.values.len() <= index {
            return None;
        } else {
            let result = self.values[index].to_double();
            if self.treats_zero_as_none && result == 0f64 {
                return None;
            } else {
                return Some(result);
            }
        }
    }
}
