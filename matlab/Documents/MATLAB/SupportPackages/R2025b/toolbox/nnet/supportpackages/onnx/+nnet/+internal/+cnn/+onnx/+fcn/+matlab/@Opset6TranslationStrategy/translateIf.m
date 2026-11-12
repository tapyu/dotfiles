function nodeTranslation = translateIf(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2025 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "else_branch"      "GRAPH"    false    []
    "then_branch"      "GRAPH"    false    []
    });
% Parse the attributes
[else_branch, then_branch] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
cond      = node.input{1};
[varCond, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, cond);
outputs   = node.output;
ThenGraph = then_branch;
ElseGraph = else_branch';
ThenGraphFcnName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'ThenGraph']);
ElseGraphFcnName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'ElseGraph']);
% Recursive calls to translate the subgraphs
ThenGraphTranslation = nnet.internal.cnn.onnx.fcn.GraphTranslation(ThenGraph, nodeTranslation.OpsetVersion, IntegerTensorNames, nodeTranslation.GraphTranslation.PackageName,...
                                                                    nodeTranslation.GraphTranslation, nodeTranslation.GraphTranslation.CoderMode, ThenGraphFcnName);
ElseGraphTranslation = nnet.internal.cnn.onnx.fcn.GraphTranslation(ElseGraph, nodeTranslation.OpsetVersion, IntegerTensorNames, nodeTranslation.GraphTranslation.PackageName,...
                                                                    nodeTranslation.GraphTranslation, nodeTranslation.GraphTranslation.CoderMode, ElseGraphFcnName);
% add the subgraph intermediate tensor names to current graph
nodeTranslation.GraphTranslation.IntermediateTensorNames = [nodeTranslation.GraphTranslation.IntermediateTensorNames, ThenGraphTranslation.IntermediateTensorNames, ElseGraphTranslation.IntermediateTensorNames];
ThenFromParentsTensor = intersect(ThenGraphTranslation.IntermediateTensorNames, ThenGraphTranslation.ParentsIntermediateTensorNames);
ElseFromParentsTensor = intersect(ElseGraphTranslation.IntermediateTensorNames, ElseGraphTranslation.ParentsIntermediateTensorNames);
if(~isempty(ThenFromParentsTensor))
    ThenGraphInputs = [ThenGraphTranslation.ExternalInputNames, cellfun(@(ch)[ch 'NumDims'], ThenGraphTranslation.ExternalInputNames, 'UniformOutput',false),...
        ThenFromParentsTensor, cellfun(@(ch)[ch 'NumDims'], ThenFromParentsTensor, 'UniformOutput',false)];
else
    ThenGraphInputs = [ThenGraphTranslation.ExternalInputNames, cellfun(@(ch)[ch 'NumDims'], ThenGraphTranslation.ExternalInputNames, 'UniformOutput',false)];
end
if(~isempty(ElseFromParentsTensor))
    ElseGraphInputs = [ElseGraphTranslation.ExternalInputNames, cellfun(@(ch)[ch 'NumDims'], ElseGraphTranslation.ExternalInputNames, 'UniformOutput',false),...
        ElseFromParentsTensor, cellfun(@(ch)[ch 'NumDims'], ElseFromParentsTensor, 'UniformOutput',false)];
else
    ElseGraphInputs = [ElseGraphTranslation.ExternalInputNames, cellfun(@(ch)[ch 'NumDims'], ElseGraphTranslation.ExternalInputNames, 'UniformOutput',false)];
end
ThenGraphInputs = strjoin(ThenGraphInputs,', ');
ElseGraphInputs = strjoin(ElseGraphInputs, ', ');
if ~isempty(ThenGraphInputs)
    ThenGraphInputs = strcat(', ',ThenGraphInputs);
end
if ~isempty(ElseGraphInputs)
    ElseGraphInputs = strcat(', ', ElseGraphInputs);
end

% Generate code for the IF operator call
[outputVars, outputRanks] = cellfun(@(s)getVarAndNumDimName(nodeTranslation.GraphTranslation, s), outputs, 'UniformOutput', false);
outputVarList = strjoin(outputVars, ', ');
outputRankList = strjoin(outputRanks, ', ');
if ~isempty(nodeTranslation.GraphTranslation.PackageName)
    % called from importNetworkFromONNX
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('if %s\n',varCond),...
        sprintf('[%s, %s] = %s(this%s, Training);\n',outputVarList,outputRankList,ThenGraphFcnName,ThenGraphInputs),...
        sprintf('else\n'),...
        sprintf('[%s, %s] = %s(this%s, Training);\n',outputVarList,outputRankList,ElseGraphFcnName,ElseGraphInputs),...
        sprintf('end\n')];
else
    % called from importONNXFunction
    % takes `Vars`, `NumDims`, `Training` and `state` inputs, dont need graph inputs
    % returns graph outputs and updated `state`
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('if %s\n',varCond),...
        sprintf('[%s, %s, state] = %s(Vars, NumDims, Training, state);\n',outputVarList,outputRankList,ThenGraphFcnName),...
        sprintf('else\n'),...
        sprintf('[%s, %s, state] = %s(Vars, NumDims, Training, state);\n',outputVarList,outputRankList,ElseGraphFcnName),...
        sprintf('end\n')];
end
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames	= [ThenGraphTranslation.IncludedFunctionNames, ElseGraphTranslation.IncludedFunctionNames];
nodeTranslation.SubgraphFunctionCode    = [ThenGraphTranslation.GraphFunctionCode, ElseGraphTranslation.GraphFunctionCode];
nodeTranslation.SubgraphTranslations    = [ThenGraphTranslation, ElseGraphTranslation];

thenTranslationIssues                   = arrayfun(@(nodeTranslation) nodeTranslation.Issues, ThenGraphTranslation.NodeTranslations, 'UniformOutput', false);
elseTranslationIssues                   = arrayfun(@(nodeTranslation) nodeTranslation.Issues, ElseGraphTranslation.NodeTranslations, 'UniformOutput', false);
nodeTranslation.Issues                  = [nodeTranslation.Issues, thenTranslationIssues{:}, elseTranslationIssues{:}];
end