function nodeTranslation = translateSlice(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2026 The MathWorks, Inc.

% Slice-10 loses its 3 attributes, which are now inputs.
% The 'steps' input has also been added.
Y       = node.output{1};
X       = node.input{1};
Starts	= node.input{2};
Ends  	= node.input{3};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarStarts, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Starts);
[VarEnds, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Ends);
% Axes and steps are optional
if numel(node.input) > 3 && ~isempty(node.input{4})
    Axes = node.input{4};
    [VarAxes, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Axes);
else
    VarAxes = "''";
end
if numel(node.input) > 4 && ~isempty(node.input{5})
    Steps = node.input{5};
    [VarSteps, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Steps);
else
    VarSteps = "''";
end

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[Indices, %s] = %sprepareSliceArgs(%s, %s, %s, %s, %s, %s);\n',...
    NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarStarts, ...
    VarEnds, VarAxes, VarSteps, NumDimX),...
    sprintf('%s = %s(Indices{:});\n', VarY, VarX),...
    ];
nodeTranslation.IncludedFunctionNames   = "prepareSliceArgs";
nodeTranslation.MCode                   = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end

end