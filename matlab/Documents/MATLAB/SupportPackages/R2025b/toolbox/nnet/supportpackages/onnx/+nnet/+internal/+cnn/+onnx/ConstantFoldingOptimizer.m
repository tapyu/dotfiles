classdef ConstantFoldingOptimizer < handle
    % Optimizes an ONNX graph by folding constants (initializers) 

    %   Copyright 2021-2024 The MathWorks, Inc.
    
    properties (SetAccess = protected, GetAccess = protected)
        Graph 
        GraphProtoManager
        Opset
        Mode
        PreserveWeightSharing = false; % This is not currently enabled as the importer does not support weight sharing. 
        TensorProducerMap
        TensorConsumerMap
        NodeGroupMap
        InitializerMap
        NewInitializers        
    end
    
    methods
        function this = ConstantFoldingOptimizer(graphProtoManager, mode)
            this.GraphProtoManager = graphProtoManager;
            this.Graph = graphProtoManager.GraphProto;
            this.Opset = graphProtoManager.OpsetVersion;
            this.Mode = mode;
            this.nameAllNodes();
            this.createMaps();
        end
        
        function graph = optimizeGraphProto(this) 
            % First, perform legacy "shallow" folding, to avoid code
            % generation for simple nodes (Constant, Reshape, Unsqueeze)
            this.NewInitializers = shallowFold(this);       
            this.Graph.initializer = [this.Graph.initializer this.NewInitializers];
            
            % Get a list of tensor names that are used in sub-graphs. These
            % will be used later when evaluating constant groups so that
            % initializers can be made for tensors that are folded away.            
            subgraphTensorReferences = getTensorsReferencedBySubgraphs(this);
            
            if isequal(this.Mode, 'deep') && this.Opset >= 6 % Functional import only supports Opsets 6
                % Update the TensorProducerMap and InitializerMap
                updateMaps(this);
                
                % Loop over the initializers, folding them one by one.
                constantGroups = arrayfun(@(init) nnet.internal.cnn.onnx.ConstantGroup(init), this.Graph.initializer);
                unusedConstGroups = [];
                for i=1:numel(constantGroups)    
                    constGroup = constantGroups(i);
                    % Inspect the next node(s) after the initializer
                    inputName = constGroup.Initializers(1).name;
                    nextNodes = [];
                    if isKey(this.TensorConsumerMap, inputName)
                        % Check whether anything actually takes this tensor as input
                        nextNodes = this.TensorConsumerMap(inputName); % Get the nodes that take the tensor as input
                    end

                    % Determine which nodes to fold
                    nodesToFold = foldableNodes(this, nextNodes);
                    while(numel(nodesToFold) > 0)
                        % Fold the last node in the list
                        nextNodeToFold = nodesToFold(end);
                        mergedConstGroups = addNodeToGroup(this, nextNodeToFold, constGroup);

                        % Update the list of intermediate constants that are discarded
                        unusedConstGroups = [unusedConstGroups mergedConstGroups];

                        % Remove the last node from the list after it has been folded
                        nodesToFold = nodesToFold(1:end-1);

                        % Determine if any of the nextNodeToFold's children can be folded.
                        % If yes, queue them to be folded next by adding them to
                        % nodesToFold
                        nextNodes = this.TensorConsumerMap(nextNodeToFold.output{1});
                        newNodesToFold = foldableNodes(this, nextNodes);
                        nodesToFold = [nodesToFold newNodesToFold];
                    end   
                end

                % Get rid of ConstantGroups that were later merged into other
                % ConstantGroups. Note: 'unique' works here because ConstantGroup is a
                % handle class.
                unusedConstGroups = unique(unusedConstGroups);
                constantGroups = setdiff(constantGroups, unusedConstGroups);

                % After folding, each ConstantFold contains groups of nodes that process
                % the starting initializers. Compute values by evaluating the operators in each
                % ConstantGroup
                for i=1:numel(constantGroups)    
                    constGroup = constantGroups(i);
                    % Only evaluate constants that have nodes in them. Existing
                    % initializers need not be updated.
                    if numel(constGroup.Nodes) > 0
                        % Compute the value
                        newInitializers = constGroup.evaluateNodes(this.Opset, this.TensorConsumerMap, subgraphTensorReferences);

                        % Update the initializer maps with the new initializers. 
                        this.NewInitializers = [this.NewInitializers newInitializers];
                        
                        % Remove the nodes consumed by the ConstantGroup
                        removeFoldedNodes(this, constGroup); 
                    end                                                       
                end
            end
            % Return updated graph with new initializers
            graph = this.Graph;
            graph.initializer = [graph.initializer this.NewInitializers];
        end
        
        function removeFoldedNodes(this, constantGroups)
            % Delete nodes from the graph that were folded into a constant
            % group.
            if numel(this.Graph.node) > 0
                % Map nodes to their indices
                nodeNames = arrayfun(@(x) string(x.name), this.Graph.node);
                nodeNameToIdxMap = containers.Map(nodeNames, 1:numel(this.Graph.node));

                % Get the names of all nodes within the constant groups
                nodeNamesToDelete = arrayfun(@(const) const.getNodeNames(), constantGroups, 'UniformOutput', false); 
                nodeNamesToDelete = [nodeNamesToDelete{:}];

                % Make sure the node wasn't already deleted
                isInIdxMap = arrayfun(@(nodeName) isKey(nodeNameToIdxMap, nodeName), nodeNamesToDelete);
                nodeNamesToDelete = nodeNamesToDelete(isInIdxMap);

                % Get unique node indices
                nodeIdxToDelete = arrayfun(@(nodeName) nodeNameToIdxMap(nodeName), nodeNamesToDelete);
                nodeIdxToDelete = unique(nodeIdxToDelete);

                this.Graph.node(nodeIdxToDelete) = [];    
            end
        end
        
        
        function [initializerNames, initializerDimMap, initializerRawDataMap] = updateInitializerMaps(this, initializerNames, initializerDimMap, initializerRawDataMap)
            newInitializerNames   = arrayfun(@(tensorProto) string(tensorProto.name), this.NewInitializers);
            newInitializerDims    = arrayfun(@(tensorProto) tensorProto.dims, this.NewInitializers, 'UniformOutput', false);
            newInitializerRawData = arrayfun(@(tensorProto) nnet.internal.cnn.onnx.getDataFromTensorProto(tensorProto), this.NewInitializers, 'UniformOutput', false);
            initializerNames         = [initializerNames, string(newInitializerNames)];
            for i=1:numel(newInitializerNames)
                initializerDimMap(newInitializerNames{i}) = newInitializerDims{i};
                initializerRawDataMap(newInitializerNames(i)) = newInitializerRawData{i};
            end            
        end
    end 
    
    methods(Access=protected)
        %% Methods used for folding
        
        function newInitializers = shallowFold(this)
            % Transform nodes which do not need to be evaluated using
            % generated code into initializers:
            % (1) Constant nodes
            % (2) Reshape nodes that take an initializer as input.
            % (3) Unsqueeze nodes that take an initializer as input.
            toDelete = [];
            newInitializers = [];
            for i = 1:numel(this.Graph.node)
                node        = this.Graph.node(i);
                op_type     = node.op_type;
                init = [];
                switch op_type
                    case 'Constant'
                        init = transformConstantNodeToInitializer(this, node);
                    case 'Reshape'
                        init = transformReshapeNodeToInitializer(this, node);
                    case 'Unsqueeze'
                        init = transformUnsqueezeNodeToInitializer(this, node);
                end
                if ~isempty(init)      
                    % Add initializer to the list of new inializers, and add 
                    % node to list of nodes to remove
                    newInitializers = [newInitializers init];
                    toDelete = [toDelete i];
                    
                    % Update the TensorConsumerMap 
                    if isKey(this.TensorConsumerMap, init.name)
                        currentConsumers = this.TensorConsumerMap(init.name);
                        currentConsumerNames = arrayfun(@(n) n.name, currentConsumers, 'UniformOutput', false);
                        currentConsumerIdxToRemove = strcmp(currentConsumerNames, node.name);
                        this.TensorConsumerMap(init.name) = currentConsumers(~currentConsumerIdxToRemove);
                    end
                    
                    % Update the InitializerMap
                    this.InitializerMap(init.name) = init;
                end
            end
            this.Graph.node(toDelete) = [];
        end
        
        function init = transformConstantNodeToInitializer(~, node)
            [rawData, dim, type, translationIssue] = nnet.internal.cnn.onnx.getDataFromConstantNode(node);
            if ~isempty(translationIssue)
                error(translationIssue.Message);
            end
            dim = double(dim);      % Note that this covers the empty case too.
            rawData = double(rawData);                                     
            init = nnet.internal.cnn.onnx.makeTensorProtoOfType(node.output{1}, dim, rawData, type);            
        end

        function init = transformReshapeNodeToInitializer(this, node)
            % If this node is reshaping an initializer, reshape it here and return it.
            inputName = node.input{1};
            if ~isKey(this.InitializerMap, inputName)
                % The first input is not an initializer
                init = [];
                return;
            else
                % The first input is an initializer.
                % Get the desired shape
                if this.Opset < 5
                    % Get the shape from an attribute
                    attributeNames      = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
                    attributeInts       = arrayfun(@(a) a.ints, node.attribute,'UniformOutput',false);
                    attributeIntsMap    = containers.Map(attributeNames, attributeInts);
                    dims                = double(attributeIntsMap('shape'));
                else
                    % Opset >= 5
                    % Shape is input 2
                    if numel(node.input)~=2 || ~isKey(this.InitializerMap, node.input{2})
                        init = [];
                        return;
                    end
                    shapeName   = node.input{2};
                    dims        = double(this.getRawData(shapeName));
                end
                if any(dims==0)
                    % If the Opset Version is 14 and plus, check allowzero
                    % parameter. If allowzero = 1, honor the 0 value, if
                    % allowzero = 0(default), infer the 0 value output dimension
                    % based on the corresponding input dimension value. 
                    allowzero = int32(0);
                    if this.Opset > 13
                        attributeNames      = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
                        attributeInt        = arrayfun(@(a) a.i, node.attribute,'UniformOutput',false);
                        attributeIntMap     = containers.Map(attributeNames, attributeInt);
                        allowzero           = int32(attributeIntMap('allowzero'));
                    end
                    if isempty(allowzero) || ~allowzero
                        % If allowzero is false or not set, replace zero value in
                        % dims with input initializer actual size
                        inputDims = this.InitializerMap(node.input{1}).dims;
                        i0 = find(dims==0);
                        dims(i0) = inputDims(i0); 
                    end
                end
                if any(dims==-1)
                    % If a -1 in the dims, infer the dims value based on
                    % the input initializer dims
                    inputDims = this.InitializerMap(node.input{1}).dims;
                    dims(dims<0) = prod(inputDims)/prod(dims(dims>0));
                end
                % Return the first input's raw data, and the desired shape
                rawData = single(this.getRawData(inputName));
                type = this.getType(inputName);
                init = nnet.internal.cnn.onnx.makeTensorProtoOfType(node.output{1}, dims, rawData, type);
            end
        end

        function init = transformUnsqueezeNodeToInitializer(this, node)
            % If this node is unsqueezing an initializer, unsqueeze it here and return it.
            inputName = node.input{1};
            if ~isKey(this.InitializerMap, inputName)
                % The input is not an initializer
                init = [];
                return;
            else
                % The input is an initializer.
                % Get axes
                if this.Opset <= 12
                    % Axes is a required attribute in Opsets 12 and earlier
                    attributeNames      = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
                    attributeInts       = arrayfun(@(a) a.ints, node.attribute,'UniformOutput',false);
                    attributeIntsMap    = containers.Map(attributeNames, attributeInts);
                    Axes                = double(attributeIntsMap('axes'));        
                else
                    % Axes is a required input starting in Opset 13
                    AxesName = node.input{2};
                    if ~isKey(this.InitializerMap, AxesName)
                        % Axes is not an initializer
                        init = [];
                        return;
                    else
                        axesInit = this.InitializerMap(AxesName);
                        rawData = nnet.internal.cnn.onnx.getDataFromTensorProto(axesInit);
                        Axes = double(rawData);
                    end
                end
                % Insert 1's into the shape
                shapeInit     = this.InitializerMap(inputName);
                shape         = shapeInit.dims;
                shapeLen      = numel(Axes) + numel(shape);
                newShape      = ones(1,shapeLen);
                Axes(Axes < 0)  = numel(shape) + Axes(Axes < 0); % Convert negative axes to positive values
                idx             = setdiff(1:shapeLen, Axes+1);
                newShape(idx)   = shape;
                % Return the input raw data, and the desired shape
                dim     = newShape;
                rawData = single(this.getRawData(inputName));
                type = this.getType(inputName);
                init = nnet.internal.cnn.onnx.makeTensorProtoOfType(node.output{1}, dim, rawData, type);
            end
        end    
        
        function updateMaps(this)
            for i=1:numel(this.NewInitializers)
                % Update producers
                this.TensorProducerMap(this.NewInitializers(i).name) =  this.NewInitializers(i);
                
                % Update initializers
                this.InitializerMap(this.NewInitializers(i).name) = this.NewInitializers(i);
            end
        end

        function tf = hasNonConstantInput(this, nextNode)
            % Does the node contain any inputs that are not derived from
            % "constants" (initializers, or other ConstantGroups)?
            tf = true;
            inputNames = nextNode.input;
            isConst = false(1, numel(inputNames));
            for i=1:numel(inputNames)
                if isKey(this.TensorProducerMap, inputNames{i})
                    inputProducer = this.TensorProducerMap(inputNames{i});
                    if isKey(this.NodeGroupMap, inputProducer.name) && ~isempty(this.NodeGroupMap(inputProducer.name))
                        isConst(i) = true;
                    elseif isa(inputProducer, 'nnet.internal.cnn.onnx.TensorProto')
                        isConst(i) = true;
                    end
                end
            end
            if all(isConst)
                tf = false;
            end
        end
        
        function tf = operatorSupported(~, node)
            supportedOperatorTypes = ["ArgMax", "Concat", "Constant", ...
                "ConstantOfShape", "Exp", "Flatten", "Gather", "Hardmax",...
                "Identity", "Log", "Max", "Min", "NonZero", "Pad",...
                "Reshape", "Round", "Scatter", "ScatterElements", "Shape",...
                "Slice", "Split", "Squeeze", "Unsqueeze"];
            tf = ismember(node.op_type, supportedOperatorTypes);
        end   
        
        function nodesToFold = foldableNodes(this, nextNodes)
            % Returns a subset of nodes from nextNodes that are eligible
            % for folding
            nodesToFold = [];
            if this.PreserveWeightSharing 
                if numel(nextNodes) == 1 && this.operatorSupported(nextNodes)
                    nodesToFold = nextNodes;
                end
            else
                for j=1:numel(nextNodes)
                    % For each next node, add it to the list of things to fold if
                    % all of its inputs are constants and its output is not
                    % dynamically referenced in any sub-graphs.
                    n = nextNodes(j);                 
                    foldable = ~hasNonConstantInput(this, n) && operatorSupported(this, n);
                    if foldable
                        % Add the node to the list of nodes to be folded
                        nodesToFold = [n nodesToFold];
                    end
                end
            end
        end

        function mergedConstantGroups = addNodeToGroup(this, node, curConstGroup)
            % Folds a single node without continuing to fold recursively.              
            mergedConstantGroups = [];
            nodeInputs = node.input;
            % Add all of the inputs to the constant group
            for i=1:numel(nodeInputs)
                inputTensorName = nodeInputs{i};
                if isKey(this.TensorProducerMap, inputTensorName)
                    inputTensorProducer = this.TensorProducerMap(inputTensorName);    
                    if isKey(this.NodeGroupMap, inputTensorProducer.name) && ~isempty(this.NodeGroupMap(inputTensorProducer.name))
                        % If the node is already a member of a ConstantGroup, merge its existing
                        % constGroup into the current constGroup. 
                        otherConstGroup = this.NodeGroupMap(inputTensorProducer.name);
                        if ~isequal(curConstGroup, otherConstGroup)
                            otherConstGroupNodeNames = otherConstGroup.getNodeNames();
                            curConstGroup.addNode(otherConstGroup.Nodes);
                            curConstGroup.addInitializer(otherConstGroup.Initializers);
                            for j=1:numel(otherConstGroupNodeNames)
                                this.NodeGroupMap(otherConstGroupNodeNames(j)) = curConstGroup;
                            end               
                            mergedConstantGroups = [mergedConstantGroups otherConstGroup];
                        end
                    end    
                end
            end
            % Add the current node to the constant group    
            curConstGroup.addNode(node);
            % If the node takes any initializers as input, add them to the
            % constant group
            addInitializersToConstantGroup(this, curConstGroup, node);

            % Update the map of nodes -> constant groups
            this.NodeGroupMap(node.name) = curConstGroup;
        end

        function addInitializersToConstantGroup(this, curConstGroup, node)
            % If any of the node's inputs are initializers, add them to the constant
            % group
            nodeInputs = node.input;
            inputProducers = cellfun(@(nodeInput) getProducerFromMap(this, nodeInput), nodeInputs, 'UniformOutput', false);
            initIdx = cellfun(@(producer) isa(producer, 'nnet.internal.cnn.onnx.TensorProto'), inputProducers);
            initsToAdd = inputProducers(initIdx);
            for i=1:numel(initsToAdd)
                curConstGroup.addInitializer(initsToAdd{i});
            end
        end

        function producer = getProducerFromMap(this, tensor)
            % Get the node that produces  the tensor
            producer = [];
            if isKey(this.TensorProducerMap, tensor)
                producer = this.TensorProducerMap(tensor);
            end
        end

        function nameAllNodes(this)
            % Assign a name to any nodes that are missing one.
            for i=1:numel(this.Graph.node)
                if isempty(this.Graph.node(i).name)
                    this.Graph.node(i).name = nnet.internal.cnn.onnx.fcn.uniqueName('node');
                end
            end
        end
        
        function createMaps(this)
            % Initialize the TensorProduerMap, TensorConsumerMap,
            % NodeGroupMap, and InitializerMap
            
            this.TensorProducerMap = containers.Map; % Tensor name -> the node or initializer that outputs it
            this.TensorConsumerMap = containers.Map; % Tensor name -> the nodes that take it as input
            this.NodeGroupMap      = containers.Map;
            this.InitializerMap    = containers.Map;

            % Add initializers to producer map and initializer map
            for i=1:numel(this.Graph.initializer)
                init = this.Graph.initializer(i);
                this.InitializerMap(init.name) = init;
                this.TensorProducerMap(init.name) = init;
            end            
            for i=1:numel(this.Graph.sparse_initializer)
                sparse_init = this.Graph.sparse_initializer(i);
                this.InitializerMap(sparse_init.values.name) = sparse_init;
                this.TensorProducerMap(sparse_init.values.name) = sparse_init;
            end

            % Add node inputs and outputs to the maps
            nodeProtos = this.Graph.node;
            for i=1:numel(nodeProtos)
                node = nodeProtos(i);
                inputs = node.input;
                outputs = node.output;
                for j=1:numel(inputs)
                    inputName = inputs{j};
                    if isKey(this.TensorConsumerMap, inputName) % Multiple nodes can take the same tensor as input
                        otherNodes = this.TensorConsumerMap(inputName);
                        this.TensorConsumerMap(inputName) = [otherNodes node];
                    else
                        this.TensorConsumerMap(inputName) = node;
                    end
                end
                for j=1:numel(outputs)
                    this.TensorProducerMap(outputs{j}) = node; % Only one node can produce a tensor with given name.
                end
            end
        end
        
        function tensorNames = getTensorsReferencedBySubgraphs(this)
            % Get subgraphs (recursively)
            subgraphs = getSubgraphs(this, this.Graph);
            tensorNames = {};
            % Get names of tensors referenced as input to a node
            for i=1:numel(subgraphs)
                tensorNames = [tensorNames, subgraphs(i).node.input];
            end
            % Remove empty names
            isEmptyName = cellfun(@isempty, tensorNames);
            tensorNames(isEmptyName) = [];
            % Get unique names
            tensorNames = unique(tensorNames);           
        end

        function subgraphs = getSubgraphs(this, graphProto)
            % Return a list of GraphProtos that are sub-graphs of nodes in
            % the top-level graph (this.Graph).
            subgraphs = [];
            for i=1:numel(graphProto.node)
                subgraphs = [subgraphs getGraphsInNode(this, graphProto.node(i))];
            end
        end
        
        function subgraphs = getGraphsInNode(this, node)
            % Extract any GraphProtos from the attributes of the node.
            % Recursively look for subgraphs in any found GraphProtos.
            subgraphs = [];
            for i = 1:numel(node.attribute)
                subgraphs = [node.attribute(i).g, node.attribute(i).graphs];
                for j = 1:numel(subgraphs)
                    subgraphs = [subgraphs, getSubgraphs(this, subgraphs(j))];
                end
            end
        end
        
        %% Methods that handle initializers
        function rawData = getRawData(this, initName)
            init = this.InitializerMap(initName);
            if isa(init, 'nnet.internal.cnn.onnx.SparseTensorProto')
                rawData = nnet.internal.cnn.onnx.getDataFromSparseTensorProto(init);
            else
                rawData = nnet.internal.cnn.onnx.getDataFromTensorProto(init);
            end
        end
        
        function dims = getDims(this, initName)
            init = this.InitializerMap(initName);
            dims = init.dims;
        end
        
        function type = getType(this, initName)
            init = this.InitializerMap(initName);
            if isa(init, 'nnet.internal.cnn.onnx.SparseTensorProto')
                type = init.values.data_type;
            else
                type = init.data_type;
            end
        end
 
     
    end
        
end