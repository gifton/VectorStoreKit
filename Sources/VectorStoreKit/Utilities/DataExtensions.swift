import Foundation

// MARK: - Data Extensions for Binary Serialization

extension Data {
    
    // MARK: - Writing Methods
    
    /// Append UInt32 in little-endian format
    mutating func appendUInt32(_ value: UInt32) {
        withUnsafeBytes(of: value.littleEndian) { bytes in
            append(contentsOf: bytes)
        }
    }
    
    /// Append UInt64 in little-endian format
    mutating func appendUInt64(_ value: UInt64) {
        withUnsafeBytes(of: value.littleEndian) { bytes in
            append(contentsOf: bytes)
        }
    }
    
    /// Append Float in little-endian format
    mutating func appendFloat(_ value: Float) {
        withUnsafeBytes(of: value.bitPattern.littleEndian) { bytes in
            append(contentsOf: bytes)
        }
    }
    
    /// Append array of Floats
    mutating func appendFloats(_ values: [Float]) {
        for value in values {
            appendFloat(value)
        }
    }
    
    /// Append string with length prefix
    mutating func appendString(_ string: String) {
        guard let data = string.data(using: .utf8) else { return }
        appendUInt32(UInt32(data.count))
        append(data)
    }
    
    /// Append boolean as single byte
    mutating func appendBool(_ value: Bool) {
        append(value ? 1 : 0)
    }
    
    // MARK: - Reading Methods
    
    /// Read UInt32 from offset in little-endian format
    func readUInt32(at offset: inout Int) throws -> UInt32 {
        guard offset + 4 <= count else {
            throw VectorStoreError.dataCorrupted("Insufficient data for UInt32 at offset \(offset)")
        }
        
        let value = self[offset..<offset+4].withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: UInt32.self).littleEndian
        }
        
        offset += 4
        return value
    }
    
    /// Read UInt64 from offset in little-endian format
    func readUInt64(at offset: inout Int) throws -> UInt64 {
        guard offset + 8 <= count else {
            throw VectorStoreError.dataCorrupted("Insufficient data for UInt64 at offset \(offset)")
        }
        
        let value = self[offset..<offset+8].withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: UInt64.self).littleEndian
        }
        
        offset += 8
        return value
    }
    
    /// Read Float from offset in little-endian format
    func readFloat(at offset: inout Int) throws -> Float {
        guard offset + 4 <= count else {
            throw VectorStoreError.dataCorrupted("Insufficient data for Float at offset \(offset)")
        }
        
        let bitPattern = self[offset..<offset+4].withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: UInt32.self).littleEndian
        }
        
        offset += 4
        return Float(bitPattern: bitPattern)
    }
    
    /// Read array of Floats from offset
    func readFloats(count floatCount: Int, at offset: inout Int) throws -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(floatCount)
        
        for _ in 0..<floatCount {
            values.append(try readFloat(at: &offset))
        }
        
        return values
    }
    
    /// Read string with length prefix from offset
    func readString(at offset: inout Int) throws -> String {
        let length = try readUInt32(at: &offset)
        
        guard offset + Int(length) <= count else {
            throw VectorStoreError.dataCorrupted("Insufficient data for String at offset \(offset)")
        }
        
        guard let string = String(data: self[offset..<offset+Int(length)], encoding: .utf8) else {
            throw VectorStoreError.dataCorrupted("Invalid UTF-8 string at offset \(offset)")
        }
        
        offset += Int(length)
        return string
    }
    
    /// Read boolean from offset
    func readBool(at offset: inout Int) throws -> Bool {
        guard offset < count else {
            throw VectorStoreError.dataCorrupted("Insufficient data for Bool at offset \(offset)")
        }
        
        let value = self[offset] != 0
        offset += 1
        return value
    }
}

// MARK: - Compression Extensions

extension Data {
    
    /// Compress data using zlib
    func compressed() throws -> Data {
        return try (self as NSData).compressed(using: .zlib) as Data
    }
    
    /// Decompress data using zlib
    func decompressed() throws -> Data {
        return try (self as NSData).decompressed(using: .zlib) as Data
    }
}

// MARK: - Checksum Extensions

extension Data {
    
    /// Calculate CRC32 checksum
    var crc32: UInt32 {
        var crc: UInt32 = 0xFFFFFFFF
        
        for byte in self {
            crc = (crc >> 8) ^ crc32Table[Int((crc ^ UInt32(byte)) & 0xFF)]
        }
        
        return ~crc
    }
    
    /// Append CRC32 checksum
    mutating func appendCRC32() {
        appendUInt32(crc32)
    }
    
    /// Verify CRC32 checksum (last 4 bytes)
    func verifyCRC32() throws -> Bool {
        guard count >= 4 else {
            throw VectorStoreError.dataCorrupted("Data too small for CRC32 verification")
        }
        
        let dataRange = startIndex..<(endIndex - 4)
        let checksumRange = (endIndex - 4)..<endIndex
        
        let expectedCRC = self[checksumRange].withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: UInt32.self).littleEndian
        }
        
        let actualCRC = self[dataRange].crc32
        
        return expectedCRC == actualCRC
    }
}

// CRC32 lookup table
private let crc32Table: [UInt32] = {
    var table = [UInt32](repeating: 0, count: 256)
    
    for i in 0..<256 {
        var c = UInt32(i)
        for _ in 0..<8 {
            if (c & 1) != 0 {
                c = 0xEDB88320 ^ (c >> 1)
            } else {
                c = c >> 1
            }
        }
        table[i] = c
    }
    
    return table
}()

// MARK: - Binary Format Helpers

extension Data {
    
    /// Create data with little-endian representation
    static func littleEndianData<T: FixedWidthInteger>(from value: T) -> Data {
        var littleEndian = value.littleEndian
        return Data(bytes: &littleEndian, count: MemoryLayout<T>.size)
    }
    
    /// Read value assuming little-endian format
    func loadLittleEndian<T: FixedWidthInteger>(as type: T.Type, at offset: Int = 0) -> T? {
        guard offset + MemoryLayout<T>.size <= count else { return nil }
        
        return self[offset..<offset + MemoryLayout<T>.size].withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: T.self).littleEndian
        }
    }
}