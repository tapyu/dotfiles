classdef TensorProto_DataType < int32
    %     UNDEFINED = 0;
    %     FLOAT = 1;   // float
    %     UINT8 = 2;   // uint8_t
    %     INT8 = 3;    // int8_t
    %     UINT16 = 4;  // uint16_t
    %     INT16 = 5;   // int16_t
    %     INT32 = 6;   // int32_t
    %     INT64 = 7;   // int64_t
    %     STRING = 8;  // string
    %     BOOL = 9;    // bool
    %     FLOAT16 = 10;
    %     DOUBLE = 11;
    %     UINT32 = 12;
    %     UINT64 = 13;
    %     COMPLEX64 = 14;     // complex with float32 real and imaginary components
    %     COMPLEX128 = 15;    // complex with float64 real and imaginary components
    %     BFLOAT16 = 16;
    %     FLOAT8E4M3FN = 17;    // float 8, mostly used for coefficients, supports nan, not inf 
    %     FLOAT8E4M3FNUZ = 18;  // float 8, mostly used for coefficients, supports nan, not inf, no negative zero 
    %     FLOAT8E5M2 = 19;      // follows IEEE 754, supports nan, inf, mostly used for gradients
    %     FLOAT8E5M2FNUZ = 20;  // follows IEEE 754, supports nan, inf, mostly used for gradients, no negative zero
    
    %   Copyright 2019-2024 The MathWorks, Inc.

    enumeration
        UNDEFINED      (0)
        FLOAT          (1)
        UINT8          (2)
        INT8           (3)
        UINT16         (4)
        INT16          (5)
        INT32          (6)
        INT64          (7)
        STRING         (8)
        BOOL           (9)
        FLOAT16        (10)
        DOUBLE         (11)
        UINT32         (12)
        UINT64         (13)
        COMPLEX64      (14)
        COMPLEX128     (15)
        BFLOAT16       (16)
        FLOAT8E4M3FN   (17)
        FLOAT8E4M3FNUZ (18)
        FLOAT8E5M2     (19)
        FLOAT8E5M2FNUZ (20) 
    end
end