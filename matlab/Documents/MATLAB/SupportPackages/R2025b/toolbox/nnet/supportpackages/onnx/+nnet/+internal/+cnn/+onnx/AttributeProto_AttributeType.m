classdef AttributeProto_AttributeType < int32
    %       enum AttributeType {
    %     UNDEFINED = 0;
    %     FLOAT = 1;
    %     INT = 2;
    %     STRING = 3;
    %     TENSOR = 4;
    %     GRAPH = 5;
    %     SPARSE_TENSOR = 11;
    %     TYPE_PROTO = 13;
    %
    %     FLOATS = 6;
    %     INTS = 7;
    %     STRINGS = 8;
    %     TENSORS = 9;
    %     GRAPHS = 10;
    %     SPARSE_TENSORS = 12;
    %     TYPE_PROTOS = 14;
    %   }

    %   Copyright 2018-2024 The MathWorks, Inc.    

    enumeration
        UNDEFINED   (0)
        FLOAT       (1)
        INT         (2)
        STRING      (3)
        TENSOR      (4)
        GRAPH       (5)
        
        FLOATS      (6)
        INTS        (7)
        STRINGS     (8)
        TENSORS     (9)
        GRAPHS      (10)
        
        SPARSE_TENSOR (11)
        SPARSE_TENSORS (12)

        TYPE_PROTO (13)
        TYPE_PROTOS (14)
    end
end
