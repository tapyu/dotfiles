function nodeTranslation = translateSum(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

% ONNX: D = Sum(A,B,C,...)
% DLT:  D = A+B+C...
D               = node.output{1};
nodeInputs      = [node.input];

numIns = numel(nodeInputs);
inputList = cell(1,numIns);
inputRankList = cell(1,numIns);
for i = 1:numIns
    [input, inputRank] = getVarAndNumDimName(nodeTranslation.GraphTranslation,node.input{i});
    inputList{i} = input;
    inputRankList{i} = inputRank;
end

[VarD, NumDimD]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, D);
nodeInputVars   = strjoin(inputList, ' + ');
nodeInputRanks  = strjoin(inputRankList, ', ');

Command = [sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = %s;\n', VarD,  nodeInputVars),...
    sprintf('%s = max([%s]);\n', NumDimD, nodeInputRanks),...
    ];
nodeTranslation.MCode = string(Command);
end

