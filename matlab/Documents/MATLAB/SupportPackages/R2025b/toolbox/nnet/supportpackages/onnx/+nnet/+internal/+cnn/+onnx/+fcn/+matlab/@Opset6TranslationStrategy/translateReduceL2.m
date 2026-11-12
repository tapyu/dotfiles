function nodeTranslation = translateReduceL2(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "axes"          "INTS"      true    []
        "keepdims"      "INT"       true    1
        });
    % Parse the attributes
    [axes, keepdims] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    Y	= node.output{1};
    X	= node.input{1};
    AxesName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Axes']);
    setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', AxesName, nnet.internal.cnn.onnx.fcn.RankedArray(axes,2));
    
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    [VarY, NumDimY]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarAxes, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, AxesName);
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('dims = %sprepareReduceArgs(%s, %s);\n', ...
        getPackageName(nodeTranslation.GraphTranslation), VarAxes, NumDimX),...
        sprintf('%s = sum(abs(%s), dims);\n', VarY, VarX)
        ];
    if keepdims
        command = [command, sprintf('%s = %s;\n', NumDimY, NumDimX)];
        nodeTranslation.IncludedFunctionNames   = "prepareReduceArgs";
    else
        command = [command, sprintf('[%s, %s] = %sonnxSqueeze(%s, %s, %s);\n', ...
            VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarY, VarAxes, NumDimX)];
        nodeTranslation.IncludedFunctionNames   = ["prepareReduceArgs", "onnxSqueeze"];
    end
    nodeTranslation.MCode                   = string(command);

    if ismember(X, IntegerTensorNames)
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    end
end