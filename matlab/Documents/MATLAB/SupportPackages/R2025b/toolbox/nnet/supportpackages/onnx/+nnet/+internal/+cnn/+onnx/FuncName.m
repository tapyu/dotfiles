classdef FuncName < int32
    
    %   Copyright 2019-2024 The MathWorks, Inc.
    enumeration
        % These fields must exactly match those of onnxmex.cpp/FuncName, in
        % order, and the numeric values here must be 0,1,...
        
        % The (de)serialize functions either save or load a
        % ModelProtoObject to/from file.
        EdeserializeFromFile        (0)
        EserializeToFile            (1)
        
        % The 'decode' functions take a pointer to an ONNX protobuf object
        % and create a MATLAB object that represents it.
        EdecodeModelProto             (2)
        EdecodeOperatorSetIdProto     (3)
        EdecodeGraphProto             (4)
        EdecodeNodeProto              (5)
        EdecodeAttributeProto         (6)
        EdecodeTensorProto            (7)
        EdecodeTensorProto_Segment    (8)
        EdecodeValueInfoProto         (9)
        EdecodeTypeProto              (10)
        EdecodeTypeProto_Map          (11)
        EdecodeTypeProto_Sequence     (12)
        EdecodeTypeProto_Tensor       (13)
        EdecodeTypeProto_Opaque       (14)    
        EdecodeTypeProto_SparseTensor (15)        
        EdecodeTypeProto_Optional     (16)
        EdecodeStringStringEntryProto (17)
        EdecodeTensorShapeProto       (18)
        EdecodeTensorShapeProto_Dimension (19)
        EdecodeSparseTensorProto      (20)
        EdecodeTensorAnnotation         (21)
        EdecodeTrainingInfoProto      (22)
        EdecodeFunctionProto          (23)
        
        % The 'encode' functions take a MATLAB object and create a C++ ONNX
        % protobuf object out of it, returning a pointer back to MATLAB.
        EnewModelProto                  (24)
        EencodeModelProto             	(25)
        EencodeOperatorSetIdProto       (26)
        EencodeGraphProto           	(27)
        EencodeNodeProto                (28)
        EencodeAttributeProto        	(29)
        EencodeTensorProto              (30)
        EencodeTensorProto_Segment      (31) 
        EencodeValueInfoProto           (32)
        EencodeTypeProto                (33)
        EencodeTypeProto_Map            (34)
        EencodeTypeProto_Sequence       (35)
        EencodeTypeProto_Tensor         (36)
        EencodeTypeProto_Opaque         (37)
        EencodeTypeProto_SparseTensor  	(38)        
        EencodeTypeProto_Optional       (39)
        EencodeStringStringEntryProto   (40)
        EencodeTensorShapeProto         (41)
        EencodeTensorShapeProto_Dimension (42)
        EencodeSparseTensorProto        (43)
        EencodeTensorAnnotation         (44)        
        EencodeTrainingInfoProto        (45)
        EencodeFunctionProto            (46)
        
        
        EdestroyModelProto              (47)
    end
end