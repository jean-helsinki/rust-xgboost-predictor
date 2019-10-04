use byteorder::{BE, LE, ReadBytesExt};

impl ModelReader for ReadBytesExt {
    #[inline]
    fn read_4bytes(&mut self) -> [u8; 4] {
        let mut buffer = [0u8; 4];
        self.read_exact(&mut buffer[..]).expect("Something went wrong reading the file");
        return buffer;
    }

    fn read_8bytes(&mut self) -> [u8; 8] {
        let mut buffer = [0u8; 8];
        self.read_exact(&mut buffer[..]).expect("Something went wrong reading the file");
        return buffer;
    }

    fn read_byte_as_int(&mut self) -> i32 {
        let mut buffer = [0u8; 4];
        self.read(&mut buffer[..1]).expect("Something went wrong reading the file");
        return i32::from_le_bytes(buffer);
    }

    fn read_int_be(&mut self) -> i32 {
        return i32::from_be_bytes(self.read_4bytes());
    }

    fn read_int_le(&mut self) -> i32 {
        return i32::from_le_bytes(self.read_4bytes());
    }

    fn read_int_array(&mut self, num_values: usize) -> Vec<i32> {
        return (0..num_values).map(|_| self.read_int_le()).collect();
    }

    fn read_uint_le(&mut self) -> u32 {
        return u32::from_le_bytes(self.read_4bytes());
    }

    fn read_long_le(&mut self) -> i64 {
        return i64::from_le_bytes(self.read_8bytes());
    }

    fn read_float_le(&mut self) -> f32 {
        return f32::from_le_bytes(self.read_4bytes());
    }

    fn read_float_array(&mut self, num_values: usize) -> Vec<f32> {
        return (0..num_values).map(|_| self.read_float_le()).collect();
    }

    fn read_double_array_be(&mut self, num_values: usize) -> Vec<f64> {
        return (0..num_values).map(|_| f64::from_be_bytes(self.read_8bytes())).collect();
    }

    fn skip(&mut self, num_bytes: usize) {
        unimplemented!()
    }

    fn read_string_with_len(&mut self, num_bytes: usize) -> String {
        unimplemented!()
    }

    fn read_string(&mut self) -> String {
        let len = self.read_long_le() as usize;
        return self.read_string_with_len(len);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let mut buf = [1u8, 0, 0, 0];
        let i = i32::from_le_bytes(buf);
        assert_eq!(i, 1);
    }
}
