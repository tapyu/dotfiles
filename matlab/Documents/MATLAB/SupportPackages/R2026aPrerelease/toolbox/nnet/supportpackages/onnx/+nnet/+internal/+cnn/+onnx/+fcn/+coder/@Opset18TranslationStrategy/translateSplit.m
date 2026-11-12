function nodeTranslation = translateSplit(~, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc. 

    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "axis"      "INT"       true    0
        "num_outputs"    "INT"   true    0
        });
    % Parse the attributes
    [axis, ~] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    numOuts = numel(node.output);
    outputList = cell(1,numOuts);
    outputRankList = cell(1,numOuts);
    for i = 1:numOuts
        [output, outputRank] = getVarAndNumDimName(nodeTranslation.GraphTranslation,node.output{i});
        outputList{i} = output;
        outputRankList{i} = outputRank;
    end
    outputListStr = strjoin(outputList, ', ');
    outputRankListStr = strjoin(outputRankList, ', ');
    X         	= node.input{1};
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    if numel(node.input)>1
        SplitName = node.input{2};
    else
        SplitName                       = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Split']);
        setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', SplitName, nnet.internal.cnn.onnx.fcn.RankedArray([], 1));
    end
    [VarSplit, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, SplitName);
    numOutputs = numel(node.output);
    code = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxSplit13(%s, %d, %s, %d, %s);\n', ...
        outputListStr, outputRankListStr, getPackageName(nodeTranslation.GraphTranslation), VarX, axis, VarSplit, numOutputs, NumDimX),...
        ];
    
    nodeTranslation.MCode                   = string(code);
    nodeTranslation.IncludedFunctionNames   = "onnxSplit13";
    if ismember(X, IntegerTensorNames)
        nodeTranslation.IntegerOutputTensorNames = string(node.output);
    end
end