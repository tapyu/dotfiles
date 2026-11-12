function nodeTranslation = translateLoop(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2025 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "body"      "GRAPH"    false    []
    });

% Parse the attributes
[body] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
M           = node.input{1};
cond        = node.input{2};

[VarM, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, M);
[VarCond, NumDimCond] = getVarAndNumDimName(nodeTranslation.GraphTranslation, cond);

v_initial	= node.input(3:end);            % "variadic. The initial values of any loop-carried dependencies (values that change across loop iterations)"
v_final_and_scan_outputs = node.output;     % "Final N loop carried dependency values then K scan_outputs"
numScanOutputs = numel(v_final_and_scan_outputs) - numel(v_initial);

% Recursive call to translate the subgraph
BodyGraphFcnName     = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'BodyGraph']);
BodyGraphTranslation = nnet.internal.cnn.onnx.fcn.GraphTranslation(body, nodeTranslation.OpsetVersion, ...
                                        IntegerTensorNames, nodeTranslation.GraphTranslation.PackageName, ...
                                        nodeTranslation.GraphTranslation, false, BodyGraphFcnName);

nodeTranslation.GraphTranslation.IntermediateTensorNames = [nodeTranslation.GraphTranslation.IntermediateTensorNames, BodyGraphTranslation.IntermediateTensorNames];

% Generate code for the Loop operator call.
%
% Inputs: According to ONNX Doc, Loop takes 2+N variadic inputs
% (iteration_num, condition, loop_carried_dependencies...).
% There are N loop carried dependencies (LCDs). Our
% loopOperator function will take 12 input arguments:
% iteration_num, condition, v_initial(a cell of LCDs), then the
% NumDims of the previous 3 args, then a handle to the "body"
% subgraph, then the full Vars and NumDims structs, Training,
% State, a handle to the subgraph function, and
% numScanOutputs(which is needed to distinguish v_final from
% scan_outputs).
%
% Outputs: According to ONNX Doc, Loop has variadic outputs
% consisting of the N LCDs followed by K "scan outputs". Our
% loopOperator function will return that, plus the NumDims of
% those variables, and the updated State. Then there must be a
% second line of code to update Vars from State.
%
%    [LCDAndScanOutputs,... LCDAndScanNumDims,... State] = loopOperator(Vars.M, Vars.cond, v_initialCell, @BodyGraphFcnName, Vars, NumDims, Training, state);
%    Vars = appendStructs(Vars, State);     % This updates Vars from State.

if ~isempty(nodeTranslation.GraphTranslation.PackageName)
    % called from importNetworkFromONNX    
    [v_initialVar, v_initialRanks] = cellfun(@(s)getVarAndNumDimName(nodeTranslation.GraphTranslation, s), v_initial, 'UniformOutput', false);
    v_initialVarList   = strjoin(v_initialVar, ', ');
    v_initialRanksList = strjoin(v_initialRanks, ', ');
    [outputVar, outputRanks] = cellfun(@(s)getVarAndNumDimName(nodeTranslation.GraphTranslation, s), v_final_and_scan_outputs, 'UniformOutput', false);
    outputVarList      = strjoin(outputVar, ', ');
    outputRanksList    = strjoin(outputRanks, ', ');

    % Handle tensors from parent graph
    BodyGraphInputsFromParent = intersect(BodyGraphTranslation.IntermediateTensorNames, BodyGraphTranslation.ParentsIntermediateTensorNames);
    BodyGraphInputsFromParentStr = "";
    if(~isempty(BodyGraphInputsFromParent))
        BodyGraphInputsFromParentNumdims = cellfun(@(ch)[ch 'NumDims'], BodyGraphInputsFromParent, 'UniformOutput',false); 
        BodyGraphInputsFromParentStr = strjoin([BodyGraphInputsFromParent BodyGraphInputsFromParentNumdims], ', ');
        BodyGraphInputsFromParentStr = strcat(BodyGraphInputsFromParentStr, ', ');
    end

    numScanOutputsStr = num2str(numScanOutputs);
    if ~isempty(VarM)
        prefixCmdM = [  sprintf('%% %s:\n', node.op_type),...        
                        sprintf('if isempty(%s)\n', VarM),...
                        sprintf('%s = Inf;\n', VarM),...        
                        sprintf('end\n')];  
    else
        VarM = nnet.internal.cnn.onnx.fcn.uniqueName('M');
        prefixCmdM = [  sprintf('%% %s:\n', node.op_type),...
                        sprintf('%s = Inf;\n', VarM)];  
    end
    if ~isempty(VarCond)
        prefixCmdCond = [   sprintf('if isempty(%s)\n', VarCond),...
                            sprintf('%s = true;\n', VarCond),...        
                            sprintf('end\n')]; 
    else
        VarCond = nnet.internal.cnn.onnx.fcn.uniqueName('Cond');
        NumDimCond = [VarCond 'NumDims'];
        prefixCmdCond = [   sprintf('%s = true;\n', VarCond),...
                            sprintf('%s = 0;\n', NumDimCond)];   
    end    
    prefixCmd = [prefixCmdM, prefixCmdCond];
    if numScanOutputs > 0
        loopCmd = [  sprintf('i = 0;\n ScanOutputs = {};\n'),...
                     sprintf('ScanOutputsOneIter = cell(1, %s);\n', numScanOutputsStr),...
                     sprintf('numDimsScanOutputsOneIter = cell(1, %s);\n', numScanOutputsStr),...
                     sprintf('while i<%s && %s\n', VarM, VarCond),...
                     sprintf('[%s, %s, ScanOutputsOneIter{:}, %s, %s, numDimsScanOutputsOneIter{:}] = this.%s(...\n', VarCond, v_initialVarList, NumDimCond, v_initialRanksList, BodyGraphFcnName),...
                     sprintf('i, %s, %s, 0, %s, %s, %s Training);\n', VarCond, v_initialVarList, NumDimCond, v_initialRanksList, BodyGraphInputsFromParentStr),...
                     sprintf('%% Update ScanOutputs \n'),...
                     sprintf('for k=1:%s\n', numScanOutputsStr),...
                     sprintf('ScanOutputs{k}{i+1} = ScanOutputsOneIter{k};\n'),...
                     sprintf('end\n i = i + 1;\n end \n'),...
                     sprintf('if isempty(ScanOutputs)\n ScanOutputs = repmat({dlarray(single([]))},1,%s);\n numDimsScanOutputs = repmat({single(0)},1, %s);\n else \n', numScanOutputsStr, numScanOutputsStr),...
                     sprintf('ScanOutputs = %sconcatenateLoopScanOutputs(ScanOutputs, numDimsScanOutputsOneIter);\n', getPackageName(nodeTranslation.GraphTranslation)),...
                     sprintf('numDimsScanOutputs = cellfun(@(R)R+1, numDimsScanOutputsOneIter, "UniformOutput", false);\n end \n'),...
                     sprintf('%% Set loop outputs \n'),...
                     sprintf('[%s] = deal(%s, ScanOutputs{:}); \n', outputVarList, v_initialVarList),...
                     sprintf('[%s] = deal(%s, numDimsScanOutputs{:}); \n', outputRanksList, v_initialRanksList)]; 

        nodeTranslation.IncludedFunctionNames	= unique(["concatenateLoopScanOutputs", BodyGraphTranslation.IncludedFunctionNames]);
    else
        loopCmd = [ sprintf('i = 0;\n'),...
                    sprintf('while i<%s && %s\n', VarM, VarCond),...
                    sprintf('[%s, %s, %s, %s] = this.%s(...\n', VarCond, v_initialVarList, NumDimCond, v_initialRanksList, BodyGraphFcnName),...
                    sprintf('i, %s, %s, 0, %s, %s, %s Training);\n i = i + 1; \n end \n', VarCond, v_initialVarList, NumDimCond, v_initialRanksList, BodyGraphInputsFromParentStr),...                    
                    sprintf('%% Set loop outputs \n'),...
                    sprintf('[%s] = [%s]; \n', outputVarList, v_initialVarList),...
                    sprintf('[%s] = [%s]; \n', outputRanksList, v_initialRanksList)];

        nodeTranslation.IncludedFunctionNames	= unique([BodyGraphTranslation.IncludedFunctionNames]);
    end
    command = [prefixCmd, loopCmd];
else
    % called from importONNXFunction
    % takes `Vars`, `NumDims`, `Training` and `state` inputs, dont need graph inputs
    % returns graph outputs and updated `state`
    if isempty(M)
        VarM = '[]';
    end
    if isempty(cond)
        VarCond = '[]';
        NumDimCond = '0';
    end
    v_initialVarList   = strjoin(cellfun(@(s)['Vars.' s], v_initial, 'UniformOutput', false), ', ');
    v_initialRanksList = strjoin(cellfun(@(s)['NumDims.' s], v_initial, 'UniformOutput', false), ', ');
    outputVarList      = strjoin(cellfun(@(s)['Vars.' s], v_final_and_scan_outputs, 'UniformOutput', false), ', ');
    outputRanksList    = strjoin(cellfun(@(s)['NumDims.' s], v_final_and_scan_outputs, 'UniformOutput', false), ', ');
    command = [...
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s, state] = onnxLoop(%s, %s, {%s}, 0, %s, {%s}, Vars, NumDims, Training, state, @%s, %d);\n', ...
        outputVarList, outputRanksList, ...
        VarM, VarCond, v_initialVarList, ...
        NumDimCond, v_initialRanksList, ...
        BodyGraphFcnName, numScanOutputs),...
        sprintf('Vars = appendStructs(Vars, state);\n')];
    nodeTranslation.InitialState = BodyGraphTranslation.InitialState;
    nodeTranslation.Nonlearnables           = nnet.internal.cnn.onnx.fcn.appendStructs(nodeTranslation.Nonlearnables, BodyGraphTranslation.Nonlearnables);
    nodeTranslation.SubgraphLearnables      = BodyGraphTranslation.Learnables;    
    nodeTranslation.IncludedFunctionNames	= unique(["onnxLoop", BodyGraphTranslation.IncludedFunctionNames]);
end
nodeTranslation.MCode                	= string(command);
nodeTranslation.SubgraphFunctionCode  	= BodyGraphTranslation.GraphFunctionCode;
nodeTranslation.SubgraphTranslations  	= BodyGraphTranslation;
% Set IntegerOutputTensorNames to be the integer outputs of the subgraph
isIntGraphOutput                        = arrayfun(@nnet.internal.cnn.onnx.isONNXTypeInteger, nodeTranslation.SubgraphTranslations.ExternalOutputTypes);
nodeTranslation.IntegerOutputTensorNames = string(nodeTranslation.SubgraphTranslations.ExternalOutputNames(isIntGraphOutput));

% Collect all translation issues
bodyTranslationIssues                   = arrayfun(@(nodeTranslation) nodeTranslation.Issues, BodyGraphTranslation.NodeTranslations, 'UniformOutput', false);
nodeTranslation.Issues                  = [nodeTranslation.Issues, bodyTranslationIssues{:}];
end

