function varargout = parseNodeAttributes(node, AttributeTable)

%   Copyright 2019 The MathWorks, Inc.

% AttributeTable column meanings: onnxName, Type, IsOptional, Default.
assert(istable(AttributeTable) && width(AttributeTable)==4, 'AttributeTable must be a table with 4 columns');
LegalTypes  = string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'));
assert(isstring(AttributeTable.(1)), 'First column of AttributeTable must contain strings');
assert(isstring(AttributeTable.(2)) && all(ismember(AttributeTable.(2), LegalTypes)), 'Second column of AttributeTable must contain legal Type strings');
assert(islogical(AttributeTable.(3)), 'Third column of AttributeTable must contain logicals');

% Get the node attributes
presentAttributeNames = string(arrayfun(@(a)string(a.name), node.attribute));  % Outer string call is to ensure it's a string even when empty.
if ~isempty(setdiff(presentAttributeNames, AttributeTable.(1)))
    if isempty(node.name) 
        nodeName = ''; 
    else
        nodeName = node.name; 
    end 
    warning(message('nnet_cnn_onnx:onnx:UnsupportedAttributes', ...
        strjoin(setdiff(presentAttributeNames, AttributeTable.(1)), ','), string(nodeName), string(node.op_type)));
end
for attrIdx = 1:height(AttributeTable)
    name        = AttributeTable{attrIdx, 1};
    type        = AttributeTable{attrIdx, 2};
    isOptional	= AttributeTable{attrIdx, 3};
    default     = AttributeTable{attrIdx, 4};
    if iscell(default)
        default = default{1};
    end
    % Set result to the default value
    varargout{attrIdx} = maybeMakeNumeric(default, type);
    % Overwrite the default value if a value is present in the node
    if ~isOptional && ~ismember(name, presentAttributeNames)
        warning(message('nnet_cnn_onnx:onnx:RequiredAttributeMissing', string(node.name), name));
        varargout{attrIdx} = [];
    elseif ~isempty(presentAttributeNames) && ismember(name, presentAttributeNames)
        varargout{attrIdx} = iGetAttrVal(node, name, type);
    end
end
end

function val = maybeMakeNumeric(val, ONNXType)
if ~isnumeric(val) && ismember(ONNXType, ["FLOAT","INT","FLOATS","INTS"])
    val = double(str2double(val));
end
end

function val = iGetAttrVal(node, name, type)
presentAttributeNames = arrayfun(@(a)string(a.name), node.attribute);
[~,idx] = ismember(name, presentAttributeNames);
switch type
    case "UNDEFINED"
        val = [];
    case "FLOAT"
        val = double(node.attribute(idx).f);
    case "INT"
        val = double(node.attribute(idx).i);
    case "STRING"
        val = string(node.attribute(idx).s);
    case "TENSOR"
        val = node.attribute(idx).t;
    case "SPARSE_TENSOR"
        val = node.attribute(idx).sparse_tensor;
    case "GRAPH"
        val = node.attribute(idx).g;
    case "FLOATS"
        val = double(node.attribute(idx).floats);
    case "INTS"
        val = double(node.attribute(idx).ints);
    case "STRINGS"
        val = string(node.attribute(idx).strings);
    case "TENSORS"
        val = node.attribute(idx).tensors;
    case "SPARSE_TENSORS"
        val = node.attribute(idx).sparse_tensors;
    case "GRAPHS"
        val = node.attribute(idx).graphs;
    otherwise
        warning(message('nnet_cnn_onnx:onnx:UnknownAttributeType', type));
        val = [];
end
end
