function nodeTranslation = translateRoiAlign(~, nodeTranslation, node, ~)                  % Introduced in opset 10
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "mode"              "STRING"	true    "avg"
    "output_height"     "INT"     	true    1
    "output_width"      "INT"       true    1
    "sampling_ratio" 	"INT"       true    0
    "spatial_scale"  	"FLOAT"     true    1.0
    });
% Parse the attributes
[mode, output_height, output_width, sampling_ratio, spatial_scale] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
if ~isequal(mode, "avg")
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node, ...
        message("nnet_cnn_onnx:onnx:UnsupportedAttributeValue", mode, "mode"));
    return;
end
% Gen code
Y               = node.output{1};
X               = node.input{1};
rois            = node.input{2};
batch_indices   = node.input{3};
[VarX, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarRois, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, rois);
[VarBatch_indices, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, batch_indices);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxRoiAlign(%s, %s, %s, %d, %d, %d, %f);\n', ...
    VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, ...
    VarRois, VarBatch_indices, output_height, output_width, sampling_ratio, spatial_scale),...
    ];
nodeTranslation.IncludedFunctionNames = "onnxRoiAlign";
nodeTranslation.MCode = string(command);
end

