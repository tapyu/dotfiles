function nodeTranslation = translateReduceSum(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "keepdims"              "INT"       true    1
    "noop_with_empty_axes"  "INT"       true    0
    });
% Parse the attributes
[keepdims, noop_with_empty_axes] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y	= node.output{1};
X	= node.input{1};
[VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumdimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
dimsName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dims');
reduceArg = nnet.internal.cnn.onnx.fcn.uniqueCoderName('xReduced'); 
if numel(node.input) > 1 && ~isempty(node.input{2})
    % Axes is specified
    AxesName = node.input{2};
    [VarAxes, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, AxesName);
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('%s = %sprepareReduceArgs(%s, %s);\n', dimsName,...
        getPackageName(nodeTranslation.GraphTranslation), VarAxes, NumdimX),...		
    ];
	if ismember(X, IntegerTensorNames)
        command = [command, sprintf('%s = fix(sum(%s, %s));\n', reduceArg, VarX, dimsName)];
	else
		command = [command, sprintf('%s = sum(%s, %s);\n', reduceArg, VarX, dimsName)];
	end
    if keepdims
        nodeTranslation.IncludedFunctionNames   = "prepareReduceArgs";
        command = [command, sprintf('%s = %s;\n', VarY, reduceArg)];
		command = [command, sprintf('%s = %s;\n', NumdimY, NumdimX)];
    else
        nodeTranslation.IncludedFunctionNames   = ["prepareReduceArgs", "onnxSqueeze"];
        command = [command, sprintf('[%s, %s] = %sonnxSqueeze(%s, %s, %s);\n', ...
            VarY, NumdimY, getPackageName(nodeTranslation.GraphTranslation), reduceArg, VarAxes, NumdimX)];
    end
else
    % Axes is empty
    if noop_with_empty_axes
        % Do nothing
        command = [
            sprintf('%% %s:\n', node.op_type),...
            sprintf('%s = %s;\n', VarY, VarX),...
            sprintf('%s = %s;\n', NumdimY, NumdimX)
            ];
    else
        % Reduce all axes
        command = [
            sprintf('%% %s:\n', node.op_type),...
            sprintf('%s = %sprepareReduceArgs([], %s);\n', dimsName, getPackageName(nodeTranslation.GraphTranslation), NumdimX),...          
            ];
		if ismember(X, IntegerTensorNames)
			command = [command, sprintf('%s = fix(sum(%s, %s));\n', reduceArg, VarX, dimsName)];
		else
			command = [command, sprintf('%s = sum(%s, %s);\n', reduceArg, VarX, dimsName)];
		end
        if keepdims
            nodeTranslation.IncludedFunctionNames   = "prepareReduceArgs";
			command = [command, sprintf('%s = %s;\n', VarY, reduceArg)];
            command = [command, sprintf('%s = %s;\n', NumdimY, NumdimX)];
        else
            nodeTranslation.IncludedFunctionNames   = ["prepareReduceArgs", "onnxSqueeze"];
            command = [command, sprintf('[%s, %s] = %sonnxSqueeze(%s, [], %s);\n', ...
                VarY, NumdimY, getPackageName(nodeTranslation.GraphTranslation), reduceArg, NumdimX)];
        end
    end
end

nodeTranslation.MCode = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

