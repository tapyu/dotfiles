function nodeTranslation = translateUnsupportedONNXLayers(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

% get inputs names
InputList = cell(1, numel(node.input));
InputNumDimsList = cell(1, numel(node.input));
for i = 1:numel(node.input)
    if ~isempty(node.input{i})
        [InputList{i},InputNumDimsList{i}] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.input{i});
    end
end
nonEmptyInputs = cellfun(@(x) ~isempty(x), InputList);
InputList = InputList(nonEmptyInputs);
InputNumDimsList = InputNumDimsList(nonEmptyInputs);
% get output names
OutputVarList = cell(1, numel(node.output));
OutputNumDimsList = cell(1, numel(node.output));
for i = 1:numel(node.output)
    if ~isempty(node.output{i})
        [OutputVarList{i},OutputNumDimsList{i}] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.output{i}); 
    end
end
nonEmptyOutputs = cellfun(@(x) ~isempty(x), OutputVarList);
OutputVarList = OutputVarList(nonEmptyOutputs);
OutputNumDimsList = OutputNumDimsList(nonEmptyOutputs);
% get attributes of this PLACEHOLDER FUNCTION
attributeHelpText = '';
VarAttrList = {};
NumDimAttrList = {};
for i = 1:numel(node.attribute)
    [value, shape, isINT] = iGetAttrValueAndShape(node.attribute(i));
    if ~isempty(value)
        % Make the value a dlarray so it could be made learnable if
        % desired. Make it double because it's an activation tensor.
        if isnumeric(value)
            value = dlarray(double(value));
        end
        
        % Construct the constant tensor at translation-time and add it to Nonlearnables
        if numel(shape)>1
            DLTShape = fliplr(shape(:)');
            value = reshape(value, DLTShape);   % Can just reshape the ONNX data because the memory ordering is the same in ONNX and DLT.
        end

        % if the attribute is {FLOAT, FLOATS, INT, INTS, TENSOR, TENSORS, SPARSE_TENSOR, SPARSE_TENSORS, STRINGS}
        AttributeName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, node.attribute(i).name]);
        setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', AttributeName, nnet.internal.cnn.onnx.fcn.RankedArray(value, numel(shape)));
        [VarAttr, NumDimAttr] = getVarAndNumDimName(nodeTranslation.GraphTranslation, AttributeName);
        % add the variable names to their corresponding list, which will be
        % passed into unsupported layer function later
        VarAttrList{end+1} = VarAttr;
        NumDimAttrList{end+1} = NumDimAttr;
        attributeHelpText = [attributeHelpText,...
            sprintf('%% "%s" of "%s" type; Value stored in "%s"; Number of dimensions stored in "%s". \n', ...
            node.attribute(i).name, string(node.attribute(i).type), VarAttr, NumDimAttr)
        ];
        if isINT
            nodeTranslation.IntegerOutputTensorNames = string(AttributeName);
        end
    else
        % if the attribute is {GRAPH, GRAPHS, UNDEFINED, TYPE_PROTO, TYPE_PROTOS}
        attributeHelpText = [attributeHelpText,...
            sprintf('%% "%s": "%s" type. \n', node.attribute(i).name, node.attribute(i).type)
        ];
    end
end

command = [sprintf('%% PLACEHOLDER FUNCTION FOR UNSUPPORTED OPERATOR (%s):\n', node.op_type)];
if ~isempty(attributeHelpText)
    % Add the attribute help text to the generated code
    command = [command, ...
        sprintf('%% "%s" has following attributes:\n', node.op_type),...
        sprintf('%s', attributeHelpText)];
end

if ~isempty(nodeTranslation.GraphTranslation.PackageName)
    % INFO call
    command = [command,...
        sprintf('[%s] = %sonnx%s(%s);\n', strjoin([OutputVarList, OutputNumDimsList], ', '),...
        getPackageName(nodeTranslation.GraphTranslation),node.op_type,strjoin([InputList, InputNumDimsList, VarAttrList, NumDimAttrList], ', '))];
else
    % IOF call
    command = [command,...
        sprintf('[%s] = PLACEHOLDER(%s);\n', strjoin([OutputVarList, OutputNumDimsList], ', '), strjoin([InputList, InputNumDimsList, VarAttrList, NumDimAttrList], ', '))];
end

nodeTranslation.MCode   = string(command);
end

function [val, shape, isINT] = iGetAttrValueAndShape(attribute)
type = attribute.type;
isINT = 0;
switch type
    case "FLOAT"
        % Get the data from a FLOAT
        val = double(attribute.f);
        shape = [];
    case "INT"
        % Get the data from a INT
        val = double(attribute.i);
        shape = [];
        isINT = 1;
    case "STRING"
        % Get the data from a STRING
        val = string(attribute.s);
        shape = [];
    case "TENSOR"
        % Get the data from a TENSOR
        value = attribute.t;
        [val, shape] = nnet.internal.cnn.onnx.getDataFromTensorProto(value);
        if nnet.internal.cnn.onnx.isONNXTypeInteger(value.values.data_type)
            isINT = 1;
        end
    case "SPARSE_TENSOR"
        % Get the data from a SparseTensorProto
        sparse_value = attribute.sparse_tensor;
        [val, shape] = nnet.internal.cnn.onnx.getDataFromSparseTensorProto(sparse_value);
        if nnet.internal.cnn.onnx.isONNXTypeInteger(sparse_value.values.data_type)
            isINT = 1;
        end
    case "FLOATS"
        % Get the data from a FLOATS
        val = double(attribute.floats);
        shape = numel(val);
    case "INTS"
        % Get the data from a INTS
        val = double(attribute.ints);
        shape = numel(val);
        isINT = 1;
    case "STRINGS"
        % Get the data from a STRINGS
        val = string(attribute.strings);
        shape = numel(val);
    case {"TENSORS", "SPARSE_TENSORS", "GRAPHS", "GRAPH", "UNDEFINED", "TYPE_PROTO", "TYPE_PROTOS"}
        % There's no operation in ONNX with "TENSORS" or "SPARSE_TENSORS" type attribute
        val = [];
        shape = [];
    otherwise
        warning(message('nnet_cnn_onnx:onnx:UnknownAttributeType', type));
        val = [];
        shape = [];
end
end