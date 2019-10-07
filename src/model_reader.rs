use byteorder::{BE, LE, ReadBytesExt};
use std::result::Result;
use std::io::Error;

pub type ModelReadResult<T> = Result<T, Error>;


pub trait ModelReader: ReadBytesExt {
    fn read_i32_le(&mut self) -> ModelReadResult<i32> {
        return self.read_i32::<LE>();
    }

    fn read_float_le(&mut self) -> ModelReadResult<f32> {
        return self.read_f32::<LE>();
    }

    fn read_byte_as_int(&mut self) -> ModelReadResult<i32> {
        let byte = self.read_u8()?;
        return Ok(byte as i32);
    }

    fn read_int_vec(&mut self, num_values: usize) -> ModelReadResult<Vec<i32>> {
        return (0..num_values).map(|_| self.read_i32_le()).collect();
    }

    fn read_float_vec(&mut self, num_values: usize) -> ModelReadResult<Vec<f32>> {
        return (0..num_values).map(|_| self.read_float_le()).collect();
    }

    fn read_double_vec_be(&mut self, num_values: usize) -> ModelReadResult<Vec<f64>> {
        return (0..num_values).map(|_| self.read_f64::<BE>()).collect();
    }

    fn skip(&mut self, num_bytes: usize) -> ModelReadResult<()> {
        let mut vec: Vec<u8> = Vec::with_capacity(num_bytes);
        return self.read_exact(vec.as_mut_slice());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {

    }
}
