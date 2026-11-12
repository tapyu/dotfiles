classdef ConstantGroup < handle
    % Handle class holding subset of nodes from a graph that can be 
    % evaluated to produce a constant value. 

    %   Copyright 2021-2024 The MathWorks, Inc.
    
    properties
        Name string
    end
    
    properties (SetAccess=protected)
        Initializers nnet.internal.cnn.onnx.TensorProto % Starting initializers that nodes in the group take as input
        Nodes nnet.internal.cnn.onnx.NodeProto % Nodes in the group
    end
    
    methods
        function this = ConstantGroup(init)
            if nargin > 0
                this.Initializers = init;                
            end
            this.Name = nnet.internal.cnn.onnx.fcn.uniqueName('constantGroup');
        end
        
        function nodes = get.Nodes(this)
            nodes = this.Nodes;
        end
        
        function initializers = get.Initializers(this)
            initializers = this.Initializers;
        end
        
        function name = get.Name(this)
            name = this.Name;
        end

        function this = addNode(this, node)
            % Add a node to the constant group
            % Accepts a single node or a list of nodes
            this.Nodes = [this.Nodes node];
        end     
        
        function this = addInitializer(this, init)
            % Add an initializer to the constant group
            % Accepts a single initializer or a list of initializers
            this.Initializers = [this.Initializers init];
        end
        
        function nodeNames = getNodeNames(this)
            % Get a string list of the node names within the constant group
            nodeNames = arrayfun(@(node) string(node.name), this.Nodes);
        end
        
        function outputName = getOutput(this)
            % Get the name of the output produced by the Constant Group
            % (the output from the group's terminal node)
            % Nodes are added in topological order, so the last node is
            % terminal.
            outputName = this.Nodes(end).output{1};
        end
        
        function newInitializers = evaluateNodes(this, opset, TensorConsumerMap, subgraphTensorReferences)
            % Evaluate the nodes in the group to produce constant value(s). 
            % Return a new initializer for each value that will be used
            % elsewhere in the graph.
            % Note: Initializers may be made for
            % intermediate tensors, not just the group's terminal node, if
            % those tensors are used as input to a node outside the group.
            
            % Procedure:
            % 1. Construct graph from the nodes
            %    1b. Make initializers into inputs
            % 2. Generate functional code for the nodes in the group
            % 3. Eval functional code using the original initializers as inputs
            % 4. Make output values into new initializers

            % Import the operator functions
            import nnet.internal.cnn.onnx.fcn.*            
            
            if isempty(this.Nodes)
                % If there are no nodes, there's nothing to generate code
                % for. Return the original initializer.
                newInitializers = this.Initializers;
            else
                %% 1. Construct a graph
                graph = makeGraphProto(this);
                
                %% 2. Generate functional code
                graphTranslation = nnet.internal.cnn.onnx.fcn.GraphTranslation(graph, opset, "", '');

                %% Create the Vars and NumDims structs
                [Vars, NumDims] = makeTensorStructs(this);

                %% 3. Evaluate the MCode in each NodeTranslation sequentially
                for i=1:numel(graphTranslation.NodeTranslations)
                    nt = graphTranslation.NodeTranslations(i);
                    code = nt.MCode;
                    for j = 1:numel(nt.IncludedFunctionNames)
                        % Prepend the op function with the support package
                        % path
                        fullPath = "nnet.internal.cnn.onnx.fcn." + nt.IncludedFunctionNames(j);
                        code = strrep(code, nt.IncludedFunctionNames(j), fullPath);
                    end
                    % Add the Node's Nonlearnables to Vars
                    nl = nt.GraphTranslation.Nonlearnables;
                    nlFields = fieldnames(nl);
                    for k = 1:numel(nlFields)
                        fname = nlFields{k};
                        Vars.(fname) = dlarray(nl.(fname).Array);
                        NumDims.(fname) = nl.(fname).Rank;
                    end 
                    % Run the code
                    evalGeneratedCode(strjoin(code, newline));
                end
                
                % Collect the final result
                resultName = this.Nodes(end).output{1};
                resultValue = extractdata(Vars.(resultName));
                resultNumDims = NumDims.(resultName);
                
                % Permute the result from reverse-ONNX dimension to 
                % forward-ONNX dimension order
                if resultNumDims > 1
                    resultValue = permute(resultValue, resultNumDims:-1:1);
                end
                
                %% 4. Make new initializers                
                % Make a new initializer for the output of the constant
                % group
                newInitializers = this.makeInitializer(graphTranslation, resultName, resultValue, resultNumDims);
                
                % Check if any of the intermediate nodes of the constant
                % group are used elsewhere in the graph               
                consumedTensors = this.tensorsConsumedOutsideConstantGroup(TensorConsumerMap, subgraphTensorReferences);
                
                % For each of those tensors, create a new initializer
                for i=1:numel(consumedTensors)
                    % Get the name and value of the tensor
                    tensorName = consumedTensors(i);
                    tensorNumDims = NumDims.(tensorName);  
                    tensorVal = Vars.(tensorName);
                    if tensorNumDims > 1
                        tensorVal = permute(tensorVal, tensorNumDims:-1:1);
                    end                    
                    
                    % Create a new initializer
                    newInit = this.makeInitializer(graphTranslation, tensorName, tensorVal, tensorNumDims);
                    newInitializers = [newInitializers newInit];
                end
            end
        end 
        
        function graphProto = removeConstantGroupNodesFromGraph(this, graphProto)
            % Removes the nodes within the constant group from the passed
            % graph and returns the modified copy.
            
            % Get the indices of the nodes to remove
            constantNodeNames = this.getNodeNames();
            idxToRemove = [];
            for i=1:numel(graphProto.node)
                idx = find(strcmp(constantNodeNames, graphProto.node(i).name));
                if ~isempty(idx)
                    idxToRemove = [idxToRemove idx];
                end
            end
            
            % Remove the nodes from the GraphProto
            graphProto.node(idxToRemove) = [];
        end
    end
    
    methods (Access=protected)
        
        function vips = makeValueInfoProtosFromInitializers(~, inits)
            % Create a ValueInfoProto for each initializer/input               
            % Set graph inputs and outputs
            vips = nnet.internal.cnn.onnx.ValueInfoProto.empty;
            for i=1:numel(inits)
                vip = nnet.internal.cnn.onnx.makeValueInfoProtoFromDimensions(...
                    inits(i).name,...
                    inits(i).data_type,...
                    num2cell(inits(i).dims));
                vips = [vips vip];
            end
        end
        
        function init = makeInitializer(~, graphTranslation, name, data, numDims)
            % Create a TensorProto of the given data and type (according to
            % graphTranslation)
            if isdlarray(data)
                data = extractdata(data);
            end
            sizeVec = size(data);
            sizeVec = int64(extendSizeVec(sizeVec, numDims)); 
            type = getTensorType(graphTranslation, name, data);
            init = nnet.internal.cnn.onnx.makeTensorProtoOfType(char(name), sizeVec, data, type);
        end
        
        function graph = makeGraphProto(this)
                % Construct a GraphProto from the nodes
                graph = nnet.internal.cnn.onnx.GraphProto;          
                graph.node = this.Nodes;
                graph.initializer = [];    

                % Make initializers into input ValueInfoProtos
                inputVIPs = makeValueInfoProtosFromInitializers(this, this.Initializers);
                graph.input = inputVIPs;
                
                % Make output ValueInfoProtos
                outputVIP = nnet.internal.cnn.onnx.makeValueInfoProtoFromDimensions(...
                    this.Nodes(end).output{1},...
                    nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT,...
                    []);
                graph.output = outputVIP;                   
        end
        
        function [Vars, NumDims] = makeTensorStructs(this)
            % Create the Vars and NumDims structs that will be used to
            % evaluate the generated functional code
            Vars = struct;
            NumDims = struct;
            for i=1:numel(this.Initializers)
                [LinearData, ONNXShape] = nnet.internal.cnn.onnx.getDataFromTensorProto(this.Initializers(i));
                if numel(ONNXShape) == 0 % OD
                    ONNXShape = [1 1];
                elseif numel(ONNXShape) == 1 % 1D
                    ONNXShape = [1 ONNXShape]; % Will be flipped to a col vec shape below
                end
                Vars.(this.Initializers(i).name) = dlarray(reshape(double(LinearData), flip(ONNXShape)));
                NumDims.(this.Initializers(i).name) = numel(this.Initializers(i).dims);
            end            
        end
        
        function tensorsConsumedOutside = tensorsConsumedOutsideConstantGroup(this, tensorConsumerMap, subgraphTensorReferences)
            % Return a string array listing the tensors produced within the
            % constant group that are used elsewhere in the main graph
            
            % List all the tensors created within the constant group
            tensorsProducedByNodes = arrayfun(@(node) string(node.output), this.Nodes, 'UniformOutput', false);
            tensorsProducedByNodes = [tensorsProducedByNodes{:}]; % A node may have multiple outputs, one cell per node
            tensorsProducedByInitializers = arrayfun(@(init) string(init.name), this.Initializers);
            allTensorsProducedByGroup = [tensorsProducedByNodes tensorsProducedByInitializers];             
            
            % Loop over the produced tensors, and check whether they are
            % consumed elsewhere
            tensorsConsumedOutside = [];
            for i=1:numel(allTensorsProducedByGroup)
                tensorName = allTensorsProducedByGroup(i);
                if ismember(tensorName, subgraphTensorReferences)
                    tensorsConsumedOutside = [tensorsConsumedOutside tensorName];
                    continue;
                end
                if isKey(tensorConsumerMap, tensorName)
                    tensorConsumers = tensorConsumerMap(tensorName);
                    % Remove the consumers that are inside the constant group
                    % from the list of tensorConsumers
                    consumersToRemoveIdx = [];
                    for j=1:numel(tensorConsumers)
                        if any(strcmp(this.getNodeNames(),  tensorConsumers(j).name))
                            consumersToRemoveIdx = [consumersToRemoveIdx, j];
                        end
                    end
                    tensorConsumers(consumersToRemoveIdx) = [];
                    % Add the tensor to the list if it is consumed elsewhere
                    if numel(tensorConsumers) > 0
                        tensorsConsumedOutside = [tensorsConsumedOutside tensorName];
                    end
                end
            end
        end
                
    end
end

function evalGeneratedCode(code)
% Evaluate the generated code in the caller's workspace (evaluateNodes())
evalin('caller', code);
end

function sz = extendSizeVec(sz, nd)
% Padthe size vector with trailing 1's until it reaches length nd
if numel(sz) < nd
    sz = [sz ones(1, nd - numel(sz))];
elseif numel(sz) > nd
    % If it's supposed to be 1d, it will be either a row or col vec in
    % MATLAB. 
    sz = max(sz);
end
end

function data = getInitializerData(graphProto)
% Return a cell array of the data for each initializer.
data = {};
for i = numel(graphProto.initializer):-1:1
    data{i} = nnet.internal.cnn.onnx.getDataFromTensorProto(graphProto.initializer(i));
    % Make all numeric data double so that input doubles can remain doubles
    % all the way through
    if isnumeric(data{i}) && ~isa(data{i}, 'double')
        data{i} = double(data{i});
    end
end
end

function data = getSparseInitializerData(graphProto)
% Return a cell array of the data for each initializer.
data = {};
for i = numel(graphProto.sparse_initializer):-1:1
    data{i} = nnet.internal.cnn.onnx.getDataFromSparseTensorProto(graphProto.initializer(i));
    if isempty(data{i})
        error(message('nnet_cnn_onnx:onnx:InitializerDataTypeUnsupported', graphProto.sparse_initializer(i).values.name,...
            char(graphProto.sparse_initializer(i).values.data_type)));
    end
end
end

function type = getTensorType(graphTranslation, tensorName, tensorData)
% Decide the type of a TensorProto. Use the GraphTranslation to check
% whether the tensor has been tracked as an integer; if so, return INT64,
% otherwise use tensorData's type
integerTensorNames = graphTranslation.IntegerTensorNames;
if ismember(tensorName, integerTensorNames)
    type = nnet.internal.cnn.onnx.TensorProto_DataType.INT64;
elseif isstring(tensorData) || ischar(tensorData)
    type = nnet.internal.cnn.onnx.TensorProto_DataType.STRING;
else
    type = nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT;
end

end