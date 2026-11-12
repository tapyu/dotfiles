classdef GraphProtoManager < handle
    % A class to hold and manipulate a single GraphProto

    %   Copyright 2021-2026 The MathWorks, Inc.

    properties (SetAccess=protected)
        GraphProto          nnet.internal.cnn.onnx.GraphProto

        OpsetVersion
    end

    properties(Access=protected)
        InitializerDimMap   % containers.Map

        InitializerRawDataMap   % containers.Map

        % a map from each tensor name to the name of the node that created
        % it (both char type)
        TensorCreatorMap % containers.Map

        % a dictionary from each tensor name to its size {string->cell}
        TensorSizeDict dictionary
    end

    properties (Dependent)
        NumNodes

        Nodes

        NodeNames

        InputVIPs

        InputNames

        ExternalInputNames

        NumOutputs

        OutputNames

        OutputVIPs

        AllInitializerNames

        % This is a cell array of nnet.internal.cnn.onnx.TensorProto and/or
        % nnet.internal.cnn.onnx.SparseTensorProto (in that order)
        AllInitializersCell cell
    end

    methods
        function this = GraphProtoManager(graphProto, opsetVersion)
            this.GraphProto = graphProto;
            this.OpsetVersion = opsetVersion;
            recreateMaps(this);
        end

        %% Getters:
        function num = get.NumNodes(this)
            num = numel(this.GraphProto.node);
        end

        function nodeArray = get.Nodes(this)
            nodeArray = this.GraphProto.node;
        end

        function nameCell = get.NodeNames(this)
            nameCell = {};
            if ~isempty(this.GraphProto.node)
                nameCell = {this.GraphProto.node.name};
            end
        end

        function inputVIPArray = get.InputVIPs(this)
            inputVIPArray = this.GraphProto.input;
        end

        function inputNameCell = get.InputNames(this)
            inputNameCell = {};
            if ~isempty(this.GraphProto.input)
                inputNameCell = {this.GraphProto.input.name};
            end
        end

        function inputNameCell = get.ExternalInputNames(this)
            inputNameCell = setdiff(this.InputNames, this.AllInitializerNames, 'stable');
        end

        function num = get.NumOutputs(this)
            num = numel(this.GraphProto.output);
        end

        function outputNameCell = get.OutputNames(this)
            outputNameCell = {};
            if ~isempty(this.GraphProto.output)
                outputNameCell = {this.GraphProto.output.name};
            end
        end

        function outputVIPs = get.OutputVIPs(this)
            outputVIPs = this.GraphProto.output;
        end

        function initArray = get.AllInitializersCell(this)
            initArray = [iArray2cell(this.GraphProto.initializer), iArray2cell(this.GraphProto.sparse_initializer)];
        end

        function nameCell = get.AllInitializerNames(this)
            % First the full initializers, then the sparse.
            fullNames = {};
            sparseNames = {};
            if ~isempty(this.GraphProto.sparse_initializer)
                values = this.GraphProto.sparse_initializer.values;
                sparseNames = {values.name};
            end
            if ~isempty(this.GraphProto.initializer)
                fullNames = {this.GraphProto.initializer.name};
            end
            nameCell = [fullNames sparseNames];
        end

        %% Questions about initializers, inputs, and outputs:
        function C = onnxInputSizeCell(this, inputTensorName)
            % Return a cell array containing the size of each dimension of
            % an ONNX input tensor. Each element can be either a number, a
            % char array, or empty. Error if the tensor is not an input
            % tensor of this graph.
            [~,pos] = ismember(inputTensorName, this.ExternalInputNames);
            assert(pos>0, message('nnet_cnn_onnx:onnx:InvalidInputTensor', inputTensorName));
            % Get the shape, including possible 'None' entries
            inputVIP = inputValueInfoProto(this, inputTensorName);
            if isempty(inputVIP.type.tensor_type.shape)
                error(message('nnet_cnn_onnx:onnx:EmptyInputDim', inputVIP.name));
            end
            dim = inputVIP.type.tensor_type.shape.dim;
            C = cell(1, numel(dim));
            if ~isempty(C)   
                C = {inputVIP.type.tensor_type.shape.dim.dim_value};          % Non-empty elements are numeric, determined sizes.
                Symbols	= {inputVIP.type.tensor_type.shape.dim.dim_param};   	% Non-empty elements indicate undetermined sizes.
                % Fill the empty numeric elements with the symbols
                emptyValuesIdx = cellfun(@(x) isempty(x), C);
                C(emptyValuesIdx) = Symbols(emptyValuesIdx);
            end
        end

        function C = onnxOutputSizeCell(this, outputTensorName)
            % Return a cell array containing the size of each dimension of
            % an ONNX output tensor. Each element can be either a number, a
            % char array, or empty. Error if the tensor is not an output
            % tensor of this graph.
            [~,pos] = ismember(outputTensorName, this.OutputNames);
            assert(pos>0, message('nnet_cnn_onnx:onnx:InvalidOutputTensor', outputTensorName));
            % Get the shape, including possible 'None' entries
            outputVIP = outputValueInfoProto(this, outputTensorName);
            if isempty(outputVIP.type) || isempty(outputVIP.type.tensor_type)
                % Only tensor outputs can be represented as a dlnetwork
                % layer output. Sequence/map/optional/etc. output types (or
                % a missing type altogether) are unsupported. Error clearly,
                % naming the output and its type, instead of crashing on the
                % missing tensor_type.
                error(message('nnet_cnn_onnx:onnx:UnsupportedModelOutputDataType', ...
                    outputVIP.name, iOutputTypeName(outputVIP.type)));
            end
            if isempty(outputVIP.type.tensor_type.shape)
                error(message('nnet_cnn_onnx:onnx:EmptyOutputDim', outputVIP.name));
            end            
            dim = outputVIP.type.tensor_type.shape.dim;
            C = cell(1, numel(dim));
            if ~isempty(C)           
                C = {outputVIP.type.tensor_type.shape.dim.dim_value};          % Non-empty elements are numeric, determined sizes.
                Symbols	= {outputVIP.type.tensor_type.shape.dim.dim_param};   	% Non-empty elements indicate undetermined sizes.
                % Fill the empty numeric elements with the symbols
                emptyValuesIdx = cellfun(@(x) isempty(x), C);
                C(emptyValuesIdx) = Symbols(emptyValuesIdx);
            end
        end

        function fullInitializers = findFullInitializers(this, initializerNameCell)
            % Find any named full initializers in initializerNameCell and
            % return the initializers in the order in which they appear in
            % the graphProto.
            fullInitializers = [];
            if ~isempty(this.GraphProto.initializer)
                [~, locs] = ismember(initializerNameCell, {this.GraphProto.initializer.name});
                locs(locs==0) = [];
                fullInitializers = this.GraphProto.initializer(locs);
            end
        end

        function sparseInitializers = findSparseInitializers(this, initializerNameCell)
            % Find any named sparse initializers in initializerNameCell and
            % return the initializers in the order in which they appear in
            % the graphProto.
            sparseInitializers = [];
            if ~isempty(this.GraphProto.sparse_initializer)
                values = this.GraphProto.sparse_initializer.values;
                [~, locs] = ismember(initializerNameCell, {values.name});
                locs(locs==0) = [];
                sparseInitializers = this.GraphProto.sparse_initializer(locs);
            end
        end

        function sz = initializerSize(this, tensorName)
            if ~isKey(this.InitializerDimMap, tensorName)
                sz = [];
            else
                sz = this.InitializerDimMap(tensorName);
            end
        end

        function data = initializerRawData(this, tensorName)
            if ~isKey(this.InitializerRawDataMap, tensorName)
                data = [];
            else
                data = this.InitializerRawDataMap(tensorName);
            end
        end

        function tf = isNodeInputInitializer(this, nodeProto, inputNums)
            tf = arrayfun(@(inputNum) numel(nodeProto.input) >= inputNum && ismember(nodeProto.input(inputNum), this.AllInitializerNames), inputNums);
        end

        function tf = isTensorInitializer(this, tensorNames)
            % vectorized
            tf = ismember(tensorNames, this.AllInitializerNames);
        end

        %% Questions about Nodes and Tensors:
        function nodeProtos = nodesFromIdx(this, indices)
            % Get nodes by indices
            nodeProtos = this.GraphProto.node(indices);
        end

        function nodeProtos = nodesFromNames(this, nodeNameCell)
            % Find the named nodes in the graph and return their
            % nodeProtos in the order in which they appear in the
            % graphProto.
            allNames = string(this.NodeNames);
            [isFound,idx] = ismember(nodeNameCell, allNames);
            assert(all(isFound));
            nodeProtos = this.Nodes(sort(idx));
        end

        function names = nodeInputNames(this, nodeNum)
            names = string(this.GraphProto.node(nodeNum).input);
        end

        function names = nodeOutputNames(this, nodeNum)
            names = string(this.GraphProto.node(nodeNum).output);
        end

        function names = allNodeInputNames(this)
            names = {};
            if ~isempty(this.Nodes)
                cellOfCells = {this.Nodes.input};
                names = unique(string(horzcat(cellOfCells{:})), 'stable');
            end
            if isempty(names)
                names = {};
            end
        end

        function names = allNodeOutputNames(this)
            names = {};
            if ~isempty(this.Nodes)
                cellOfCells = {this.Nodes.output};
                names = unique(string(horzcat(cellOfCells{:})), 'stable');
            end
            if isempty(names)
                names = {};
            end
        end

        function nodeName = tensorCreator(this, tensorName)
            if isKey(this.TensorCreatorMap, tensorName)
                nodeName = this.TensorCreatorMap(tensorName);
            else
                nodeName = '';
            end
        end

        function vip = inputValueInfoProto(this, tensorName)
            vip = [];
            for i=1:numel(this.GraphProto.input)
                if strcmp(this.GraphProto.input(i).name, tensorName)
                    vip = this.GraphProto.input(i);
                end
            end
        end

        function vip = outputValueInfoProto(this, tensorName)
            vip = [];
            for i=1:numel(this.GraphProto.output)
                if strcmp(this.GraphProto.output(i).name, tensorName)
                    vip = this.GraphProto.output(i);
                end
            end
        end

        function tensorNames = dynamicallyReferencedTensorNamesInNode(this, nodeProto, candidates)
            % Dynamically-referenced tensors in subgraphs of a node
            tensorNames = string.empty;
            for attrNum = 1:numel(nodeProto.attribute)
                g = nodeProto.attribute(attrNum).g;
                if ~isempty(g)
                    tensorNames = [tensorNames, dynamicallyReferencedTensorNamesInSubgraph(this, g, candidates)];
                end
                for gnum = 1:numel(nodeProto.attribute(attrNum).graphs)
                    g = nodeProto.attribute(attrNum).graphs(gnum);
                    tensorNames = [tensorNames, dynamicallyReferencedTensorNamesInSubgraph(this, g, candidates)];
                end
            end
        end

        function sz = tensorSize(this, tensorName)
            if isConfigured(this.TensorSizeDict) && isKey(this.TensorSizeDict, tensorName)
                sz = this.TensorSizeDict{tensorName};
            else
                sz = -1;
            end
        end

        function sz = variableSize(this, tensorName)
            if isTensorInitializer(this, tensorName)
                sz = initializerSize(this, tensorName);
            elseif ismember(tensorName, this.ExternalInputNames)
                inputSize = this.onnxInputSizeCell(tensorName);
                inputSize(cellfun(@iIsInvalidSize, inputSize)) = {NaN};
                sz = cell2mat(inputSize);
            elseif ismember(tensorName, this.OutputNames)
                outputSize = this.onnxOutputSizeCell(tensorName);
                outputSize(cellfun(@iIsInvalidSize, outputSize)) = {NaN};
                sz = cell2mat(outputSize);
            else
                sz = tensorSize(this, tensorName);
            end
        end

        %% Modifying the graph
        function foldConstants(this, foldConstants)
            % Fold constants if the user has not specified 'none'
            if ismember(foldConstants, ["shallow", "deep"])
                % Fold constants to the extent possible and update the initializer maps
                cfOpt = nnet.internal.cnn.onnx.ConstantFoldingOptimizer(this, foldConstants);
                this.GraphProto = cfOpt.optimizeGraphProto();
                recreateMaps(this);
            end            
        end

        function makeGraphNamesDLTCompatible(this, maxNameLength)
            makeNodeNamesDLTCompatible(this);
            this.GraphProto = nnet.internal.cnn.onnx.fcn.makeNamesDLTCompatible(this.GraphProto, maxNameLength);
            % Nodes and tensors have been renamed, so we need to recreate
            % the maps
            recreateMaps(this);
        end

        %% Making MATLAB digraph objects from the graphProto:
        function nodeDigraph = graphProtoToDigraph(this)
            % s and t are arguments to digraph, used to define the edges. There is an
            % edge for every instance where a tensor passes from one node to another.
            nodeSrc = string.empty;
            nodeDest = string.empty;
            candidates = unique([this.ExternalInputNames allNodeOutputNames(this)], 'stable');
            for i = 1:numel(this.GraphProto.node)
                nodeProto = this.GraphProto.node(i);
                % Add this node's input connections to the digraph connections
                for j = 1:numel(nodeProto.input)
                    inputTensorName = nodeProto.input{j};
                    % To the node graph, add a connection from the source node to the
                    % destination node
                    if isKey(this.TensorCreatorMap, inputTensorName)
                        nodeSrc(end+1) = string(this.TensorCreatorMap(inputTensorName));
                        nodeDest(end+1) = string(nodeProto.name);
                    end
                end
                % Find the dynamically referenced tensors in the subgraphs
                dynamicallyReferencedTensorNames = dynamicallyReferencedTensorNamesInNode(this, nodeProto, candidates);
                for j = 1:numel(dynamicallyReferencedTensorNames)
                    creator = tensorCreator(this, dynamicallyReferencedTensorNames{j});
                    if ~isempty(creator)...
                            &&(~isequal(string(creator),string(nodeProto.name)))
                        nodeSrc(end+1) = string(creator);
                        nodeDest(end+1) = string(nodeProto.name);
                    end
                end
            end
            allNodeNames = string({this.GraphProto.node.name});
            [~, nodeDigraph] = toposort(digraph(nodeSrc, nodeDest, 1, allNodeNames));
        end
    end

    methods(Access=protected)
        function tensorNames = dynamicallyReferencedTensorNamesInSubgraph(this, graphProto, candidates)
            % Find members of 'candidates' that are referenced dynamically
            % in this graph, taking account of shadowing by local
            % variables. Remove the graph's newly introduced local
            % variables from candidates. These are the graph inputs,
            % initializers, and all node outputs
            gpm = nnet.internal.cnn.onnx.GraphProtoManager(graphProto, this.OpsetVersion);
            newLocalTensors = [gpm.InputNames, gpm.AllInitializerNames, allNodeOutputNames(gpm)];
            candidates = setdiff(candidates, newLocalTensors, 'stable');
            % Check for nonlocal references in node inputs and graph outputs
            allReferences = unique([allNodeInputNames(gpm), gpm.OutputNames]);
            nonlocalReferences = setdiff(allReferences, newLocalTensors);
            % Find our nonlocal refs among the candidates
            tensorNames = intersect(candidates, nonlocalReferences);
            % Recurse into our own nodes to find more nonlocal dynamic refs
            for nodeNum = 1:numel(graphProto.node)
                tensorNames = unique([tensorNames, dynamicallyReferencedTensorNamesInNode(this, graphProto.node(nodeNum), candidates)]);
            end
        end

        function recreateMaps(this)
            % Recreate the initializer maps and tensor creator map.
            createInitializerMaps(this);
            createTensorCreatorMap(this);
            createTensorSizeDict(this);
        end

        function createTensorSizeDict(this)
            value_info = this.GraphProto.value_info;
            if ~isempty(value_info)
                keys = string({value_info.name});
                values = cellfun(@iGetSize,{value_info.type},'UniformOutput',false);
                this.TensorSizeDict = dictionary(keys,values);
            end
        end

        function createInitializerMaps(this)
            % Get initializers
            initializerNames    = this.AllInitializerNames;
            initializerDims     = cellfun(@(tp) tp.dims, this.AllInitializersCell, 'UniformOutput', false);
            sparseIdx           = cellfun(@(tp) isa(tp, 'nnet.internal.cnn.onnx.SparseTensorProto'), this.AllInitializersCell);
            sparseRawData       = cellfun(@(tp) nnet.internal.cnn.onnx.getDataFromSparseTensorProto(tp), this.AllInitializersCell(sparseIdx), 'UniformOutput', false);
            fullRawData         = cellfun(@(tp) nnet.internal.cnn.onnx.getDataFromTensorProto(tp), this.AllInitializersCell(~sparseIdx), 'UniformOutput', false);
            initializerRawData  = [sparseRawData fullRawData];
            if ~isempty(initializerNames)
                initializerDimMap       = containers.Map(initializerNames,initializerDims);
                initializerRawDataMap   = containers.Map(initializerNames,initializerRawData);
                clear initializerRawData    %save memeory
            else
                initializerDimMap       = containers.Map;
                initializerRawDataMap   = containers.Map;
            end
            this.InitializerDimMap      = initializerDimMap;
            this.InitializerRawDataMap  = initializerRawDataMap;
        end

        function createTensorCreatorMap(this)
            % this.TensorCreatorMap is a map from each tensor name to the name of the node
            % that created it (both char type)
            this.TensorCreatorMap = containers.Map;
            for i = 1:numel(this.GraphProto.node)
                nodeProto = this.GraphProto.node(i);
                % Add this node's outputs to this.TensorCreatorMap
                for j = 1:numel(nodeProto.output)
                    if ~isempty(nodeProto.output{j})
                        this.TensorCreatorMap(nodeProto.output{j}) = nodeProto.name;
                    end
                end
            end
        end

        function makeNodeNamesDLTCompatible(this)
            % Fill in empty names. If existing names have slashes, try to preserve
            % that information by replacing slashes with vertical bars.
            nodes = this.GraphProto.node;
            for i=1:numel(nodes)
                if isempty(nodes(i).name)
                    nodes(i).name = strcat(nodes(i).op_type, 'Node', num2str(i));
                else
                    nodes(i).name = strrep(nodes(i).name, '/', '|');
                    %         nodes(i).name = strcat(nodes(i).op_type, '_', strrep(nodes(i).name, '/', '|'));    % Also prepend the optype
                end
            end
            this.GraphProto.node = nodes;
        end

        function [tf, dim, rawData, tensorProto] = transformConstantNodeToInitializer(~, node)
            %Get the tensor attributes
            attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
            attributeTensors = arrayfun(@(a) a.t, node.attribute,'UniformOutput',false);
            attributeTensorsMap = containers.Map(attributeNames, attributeTensors);
            tf = false; dim = []; rawData = []; tensorProto = [];
            if ismember('value', attributeNames)
                tf = true;
                tproto = attributeTensorsMap('value');
                if isempty(tproto.dims)
                    dim = 1;
                else
                    dim = double(tproto.dims);
                end
                rawData = double(nnet.internal.cnn.onnx.getDataFromTensorProto(tproto));
                name  = node.output{1};
                tensorProto = tproto;
                tensorProto.name = name;               
            end
        end

        function [tf, dim, rawData, tensorProto] = transformReshapeNodeToInitializer(this, node)
            % If this node is reshaping an initializer, reshape it here and return it.
            inputName = node.input{1};
            tf = false;
            dim = [];
            rawData = [];
            tensorProto = [];
            if isKey(this.InitializerDimMap, inputName)
                % The first input is an initializer.
                % Get the desired shape
                if this.OpsetVersion < 5
                    % Get the shape from an attribute
                    attributeNames      = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
                    attributeInts       = arrayfun(@(a) a.ints, node.attribute,'UniformOutput',false);
                    attributeIntsMap    = containers.Map(attributeNames, attributeInts);
                    shape               = double(attributeIntsMap('shape'));
                else
                    % Opset >= 5
                    % Shape is input 2                    
                    shapeName   = node.input{2};                   
                    if ~isKey(this.InitializerDimMap, shapeName)
                        % Shape is not an initializer
                        return;
                    end
                    shape       = double((this.InitializerRawDataMap(shapeName)));
                end
                % Return the first input's raw data, and the desired shape
                tf = true;
                dim = shape;
                rawData = single(this.InitializerRawDataMap(inputName));
                name  = node.output{1};
                % Find the initializer
                tensorProto = initializerFromName(this, inputName);
                tensorProto.dims = int64(dim);
                tensorProto.name = name;
            end
        end

        function [tf, dim, rawData, tensorProto] = transformUnsqueezeNodeToInitializer(this, node)
            % If this node is unsqueezing an initializer, unsqueeze it here and return it.
            inputName = node.input{1};
            if ~isKey(this.InitializerDimMap, inputName)
                % The input is not an initializer
                tf = false;
                dim = [];
                rawData = [];
                tensorProto = [];
            else
                % The input is an initializer.
                % Get axes
                attributeNames      = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
                attributeInts       = arrayfun(@(a) a.ints, node.attribute,'UniformOutput',false);
                attributeIntsMap    = containers.Map(attributeNames, attributeInts);
                Axes                = double(attributeIntsMap('axes'));
                % Insert 1's into the shape
                shape         = this.InitializerDimMap(inputName);
                shapeLen      = numel(Axes) + numel(shape);
                newShape      = ones(1,shapeLen);
                idx           = setdiff(1:shapeLen, Axes+1);
                newShape(idx) = shape;
                % Return the input raw data, and the desired shape
                tf      = true;
                dim     = newShape;
                rawData = single(this.InitializerRawDataMap(inputName));

                name  = node.output{1};
                % Find the initializer
                tensorProto = initializerFromName(this, inputName);
                tensorProto.dims = int64(dim);
                tensorProto.name = name;
            end
        end

        function tensorProto = initializerFromName(this, name)
            graphProto = this.GraphProto;
            isSparse = false;
            if ~isempty(graphProto.sparse_initializer)
                values = graphProto.sparse_initializer.values;
                [isSparse,idx] = ismember(name, {values.name});
            end
            if isSparse
                tensorProto = graphProto.sparse_initializer(idx);
            else
                [~,idx] = ismember(name, {graphProto.initializer.name});
                tensorProto = graphProto.initializer(idx);
            end
        end

    end
end

function C = iArray2cell(A)
C = arrayfun(@(x)x, A, 'UniformOutput', false);
end

function tensorProto = iMakeTensorProto(name, data_type, raw_data, dims)
% Create a tensorProto from fields taken directly out of another
% tensorProto or other raw onnx structure. raw_data and dims are therefore
% to be interpreted as row-major.
tensorProto              = nnet.internal.cnn.onnx.TensorProto;
tensorProto.name         = name;
tensorProto.data_type    = data_type;
tensorProto.raw_data     = raw_data;
tensorProto.dims         = int64(dims);
end

function tf = iIsInvalidSize(x)
tf = isempty(x) || ~isnumeric(x) || (isnumeric(x) && x <= 0);
end

function size = iGetSize(type)
size = [];
if ~isempty(type.tensor_type) && ~isempty(type.tensor_type.shape)
    for i = 1:numel(type.tensor_type.shape.dim)
        if ~isempty(type.tensor_type.shape.dim(i).dim_value)
            % fixed size. Represented by numbers.
            size(end+1) = type.tensor_type.shape.dim(i).dim_value;
        else
            % dynamic size. Represented by a string.
            size(end+1) = NaN;
        end
    end
end
if isempty(size)
    size = -1;
end
end

function name = iOutputTypeName(type)
% Return the name of the ONNX output TypeProto's set type, i.e. the
% populated field of the "value" oneof (tensor_type, sequence_type, ...).
% Used to report unsupported non-tensor output types. Any new oneof field
% added to TypeProto is picked up automatically; only 'denotation' is
% skipped since it is a plain string sibling, not part of the oneof.
name = "unknown";
if isempty(type)
    return
end
typeFields = string(fieldnames(type));
typeFields(typeFields == "denotation") = [];
for i = 1:numel(typeFields)
    fName = typeFields(i);
    if ~isempty(type.(fName))
        % Turn e.g. "sequence_type" into "sequence" for the message.
        name = erase(fName, "_type");
    end
end
end