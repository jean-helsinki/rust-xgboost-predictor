use byteorder::{BE, LE, ReadBytesExt};
use std::result::Result;
use std::io::Error;

pub type ModelReadResult<T> = Result<T, Error>;


pub trait ModelReader: ReadBytesExt {
    #[inline]
    fn read_i32_le(&mut self) -> ModelReadResult<i32> {
        return self.read_i32::<LE>();
    }

    #[inline]
    fn read_f32_le(&mut self) -> ModelReadResult<f32> {
        return self.read_f32::<LE>();
    }

    #[inline]
    fn read_byte_as_i32(&mut self) -> ModelReadResult<i32> {
        let byte = self.read_u8()?;
        return Ok(byte as i32);
    }

    fn read_to_i32_buffer(&mut self, buffer: &mut [i32]) -> ModelReadResult<()> {
        for b in buffer.iter_mut() {
            match self.read_i32_le() {
                Ok(val) => {*b = val},
                Err(e) => {return Err(e)},
            }
        }
        return Ok(());
    }

    fn read_to_f32_buffer(&mut self, buffer: &mut [f32]) -> ModelReadResult<()> {
        for b in buffer.iter_mut() {
            match self.read_f32_le() {
                Ok(val) => {*b = val},
                Err(e) => {return Err(e)},
            }
        }
        return Ok(());
    }
    fn read_to_f64_buffer_be(&mut self, buffer: &mut [f64]) -> ModelReadResult<()> {
        for b in buffer.iter_mut() {
            match self.read_f64::<BE>() {
                Ok(val) => {*b = val},
                Err(e) => {return Err(e)},
            }
        }
        return Ok(());
    }

    fn read_int_vec(&mut self, num_values: usize) -> ModelReadResult<Vec<i32>> {
        return (0..num_values).map(|_| self.read_i32_le()).collect();
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
