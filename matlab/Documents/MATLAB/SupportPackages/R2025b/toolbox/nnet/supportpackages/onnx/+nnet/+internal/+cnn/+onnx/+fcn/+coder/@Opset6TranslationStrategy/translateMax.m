function nodeTranslation = translateMax(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

% ONNX: D = Max(A,B,C,...)
D               = node.output{1};
nodeInputs      = [node.input];
[relabeledInputs,relabeledInputRanks] = cellfun(@(s)getVarAndNumDimName(nodeTranslation.GraphTranslation,s), nodeInputs, 'UniformOutput', false);
[VarD, NumDimD]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, D);
nodeInputVars   = strjoin(relabeledInputs, ', ');
nodeInputRanks  = strjoin(relabeledInputRanks, ', ');
Command = [sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxMax({%s}, [%s]);\n', VarD, NumDimD, ...
    getPackageName(nodeTranslation.GraphTranslation), nodeInputVars, nodeInputRanks),...
    ];
nodeTranslation.MCode = string(Command);
nodeTranslation.IncludedFunctionNames = "onnxMax";
if ismember(nodeInputs{1}, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(D);
end
end

