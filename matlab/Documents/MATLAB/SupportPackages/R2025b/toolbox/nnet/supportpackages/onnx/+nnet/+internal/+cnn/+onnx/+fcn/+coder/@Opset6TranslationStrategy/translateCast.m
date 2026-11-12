function nodeTranslation = translateCast(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "to"         "INT"       false    []
    });
% Parse the attributes
[to] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y = node.output{1};
X = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

command = sprintf('%% %s:\n', node.op_type);
% From TensorProto, 'to' can be:
%     UNDEFINED = 0;
%     // Basic types.
%     FLOAT = 1;   // float
%     UINT8 = 2;   // uint8_t
%     INT8 = 3;    // int8_t
%     UINT16 = 4;  // uint16_t
%     INT16 = 5;   // int16_t
%     INT32 = 6;   // int32_t
%     INT64 = 7;   // int64_t
%     STRING = 8;  // string
%     BOOL = 9;    // bool
%
%     // IEEE754 half-precision floating-point format (16 bits wide).
%     // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
%     FLOAT16 = 10;
%
%     DOUBLE = 11;
%     UINT32 = 12;
%     UINT64 = 13;
%     COMPLEX64 = 14;     // complex with float32 real and imaginary components
%     COMPLEX128 = 15;    // complex with float64 real and imaginary components
%
%     // Non-IEEE floating-point format based on IEEE754 single-precision
%     // floating-point number truncated to 16 bits.
%     // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
%     BFLOAT16 = 16;

% When casting to INTs, allow truncation, then convert back to
% the original MATLAB type. Cast complex to double.
switch to
    case {1, 10, 16}
        command = [command, sprintf('%s = single(%s);\n', varY, varX)];
    case 2
        command = [command, sprintf('%s = cast(uint8(%sextractIfDlarray(%s)), ''like'', %s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), varX, varX)];
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    case 3
        command = [command, sprintf('%s = cast(int8(%sextractIfDlarray(%s)), ''like'', %s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), varX, varX)];
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    case 4
        command = [command, sprintf('%s = cast(uint16(%sextractIfDlarray(%s)), ''like'', %s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), varX, varX)];
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    case 5
        command = [command, sprintf('%s = cast(int16(%sextractIfDlarray(%s)), ''like'', %s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), varX, varX)];
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    case 6
        command = [command, sprintf('%s = cast(int32(%sextractIfDlarray(%s)), ''like'', %s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), varX, varX)];
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    case 7
        command = [command, sprintf('%s = cast(int64(%sextractIfDlarray(%s)), ''like'', %s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), varX, varX)];
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    case 8
        command = [command, sprintf('%s = string(%s);\n', varY, varX)];
    case 9
        command = [command, sprintf('%s = logical(%s);\n', varY, varX)];
    case {11, 14, 15}
        command = [command, sprintf('%s = double(%s);\n', varY, varX)];
    case 12
        command = [command, sprintf('%s = cast(uint32(%sextractIfDlarray(%s)), ''like'', %s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), varX, varX)];
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    case 13
        command = [command, sprintf('%s = cast(uint64(%sextractIfDlarray(%s)), ''like'', %s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), varX, varX)];
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    otherwise
        assert(false);
end
command = [command, sprintf('%s = %s;\n', numDimsY, numDimsX)];
nodeTranslation.MCode = string(command);
end

