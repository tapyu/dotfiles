classdef CustomLayerManager < handle

    % Copyright 2021-2026 The MathWorks, Inc.

    properties (SetAccess=protected)
        %% Graph-related properties

        % The top-level graph of the model
        TopLevelGraphProtoManager nnet.internal.cnn.onnx.GraphProtoManager

        % The full graph as digraphs
        NodeDigraph                     % digraph nodes are ONNX nodes

        %% Tensor-labeling properties

        % An object containing ONNX tensor labels and the names of fully-supported nodes
        OnnxTensorLabeler nnet.internal.cnn.onnx.ONNXTensorLabeler

        %% Custom layer-related properties

        % Package name in which to save custom layer classdef files
        PackageName char
        OrigPackageName char

        % An array of the names of the generated custom layer classes
        CustomLayerNames string

        % The input DLT format strings of the generated
        % custom layer classes. A cell array of string arrays.
        CustomLayerDLTInputFormats cell

        % The output DLT format strings of the generated custom layer
        % classes. A cell array of string arrays.
        CustomLayerDLTOutputFormats cell

        % A map from node names to custom layer names
        NodeToCustomLayerMap % containers.Map

        % An array of ModelProto objects, one for each custom layer
        ModelProtos nnet.internal.cnn.onnx.ModelProto

        % An array of ModelTranslation objects, one for each custom layer
        ModelTranslations nnet.internal.cnn.onnx.fcn.ModelTranslation
        
        % FullySupportedNodeNames
        FullySupportedNodeNames string

        % Whether any custom layers were created by splitting a concave
        % connected component, causing one layer to pass dlarrays to the
        % other
        AnyCoupledCustomLayers logical = false

        % How to handle grouping of nodes when creating custom layers.
        CustomLayerContent string

        %% Other properties
        OpsetVersion double
    end

    methods
        function this = CustomLayerManager(graphProtoManager, opsetVersion, ...
                onnxTensorLabeler, fullySupportedNodeNames, packageName, ...
                origPackageName, CustomLayerContent, generateCoderFiles, isCoderClass)
            this.TopLevelGraphProtoManager = graphProtoManager;
            this.OpsetVersion           = opsetVersion;
            this.OnnxTensorLabeler      = onnxTensorLabeler;
            this.NodeToCustomLayerMap   = containers.Map;
            this.FullySupportedNodeNames = fullySupportedNodeNames;
            this.PackageName            = packageName;
            this.OrigPackageName        = origPackageName;
            this.NodeDigraph            = graphProtoToDigraph(graphProtoManager);
            this.CustomLayerContent        = CustomLayerContent;

            customDigraphs = segmentUnsupportedSubgraphs(this);
            if ~isempty(customDigraphs) && ~isCoderClass
                warnIfPackageNameChanged(this);
            end
            dynamicReferencesCell = findNonlocalDynamicTensorReferences(this, customDigraphs);
            coreFunctions = [];
            for i=1:numel(customDigraphs)
                nodeNames = string(customDigraphs{i}.Nodes.Name);
                nameStem = makeCustomLayerName(this, nodeNames);
                this.ModelProtos(i) = subgraphToModelProto(this, nodeNames, nameStem, dynamicReferencesCell{i}, dynamicReferencesCell);
                this.ModelTranslations(i) = genCustomLayerFunction(this, this.ModelProtos(i), nameStem, isCoderClass);
                
                coreFunctions = [coreFunctions,...
                    this.ModelTranslations(i).ToplevelGraphTranslation.IncludedFunctionNames];
                
                [this.CustomLayerNames(i), this.CustomLayerDLTInputFormats{i}, ...
                    this.CustomLayerDLTOutputFormats{i}, classFileCode] = genCustomLayerClass(this, this.ModelTranslations(i), nameStem, generateCoderFiles, isCoderClass);                
                addToCustomLayerMap(this, nodeNames, this.CustomLayerNames(i));
                % Write the classdef file.
                writeClassdefFile(this, this.CustomLayerNames(i), classFileCode, isCoderClass);
            end
            if ~isempty(customDigraphs)
                % Write functions
                writeFunctions(this, coreFunctions, isCoderClass);
            end
            % Rehash to make sure newly-written classdefs are found
            rehash
        end

        function warnIfPackageNameChanged(this)
            if ~isequal(this.PackageName, this.OrigPackageName)
                warning(message('nnet_cnn_onnx:onnx:InvalidPackageName', this.OrigPackageName, this.PackageName))
            end
        end

        function cellOfCellstrs = findNonlocalDynamicTensorReferences(this, customDigraphs)
            % For each component, find tensor names that are referenced
            % from inside the component's nested graphs via dynamic
            % scoping, but which are not created by this component. (Later,
            % these tensors will be made inputs to the custom layer for
            % this component.) The candidates in the toplevel graph are all
            % graph inputs and all node output tensors. Initializers aren't
            % candidates, even if they're dynamically referenced, because
            % they're known at translation-time. So they can be put right
            % into the network at translation time and don't need to be
            % passed around at runtime
            cellOfCellstrs = {};
            graphProtoManager = this.TopLevelGraphProtoManager;
            candidates = unique([graphProtoManager.ExternalInputNames allNodeOutputNames(graphProtoManager)], 'stable');
            for i=1:numel(customDigraphs)
                subgraphNodeNames = string(customDigraphs{i}.Nodes.Name);
                subgraphNodeProtos = nodesFromNames(graphProtoManager, subgraphNodeNames);
                dynamicRefsForDigraph = string.empty;
                for nodeNum = 1:numel(subgraphNodeProtos)
                    dynamicRefsForDigraph = [dynamicRefsForDigraph, dynamicallyReferencedTensorNamesInNode(graphProtoManager, subgraphNodeProtos(nodeNum), candidates)];
                end
                % Now, from among the dynamic references, find the nonlocal
                % ones (tensors not created in, or input to, this digraph).
                % We remove all output tensors in this subgraph, because
                % they were locally created. We also remove all inputs,
                % because those will already have been made available to
                % this custom layer.
                allSubgraphInputNames = unique([subgraphNodeProtos.input], 'stable');
                allSubgraphOutputNames = unique([subgraphNodeProtos.output], 'stable');
                nonlocalDynamicRefs = setdiff(dynamicRefsForDigraph, [allSubgraphInputNames, allSubgraphOutputNames], 'stable');
                % Return that list for this subgraph
                cellOfCellstrs{i} = cellstr(nonlocalDynamicRefs);
            end
        end

        function nameStem = makeCustomLayerName(this, customLayerNodeNames)
            if numel(customLayerNodeNames)<2
                firstNodeName = customLayerNodeNames(1);
                firstNode = nodesFromNames(this.TopLevelGraphProtoManager, firstNodeName);
                firstNodeOpType = firstNode.op_type;
                nameStem = firstNodeOpType;
            else
                firstAndLastNodeNames = [customLayerNodeNames(1) customLayerNodeNames(end)];
                firstAndLastNodes = nodesFromNames(this.TopLevelGraphProtoManager, firstAndLastNodeNames);
                firstNodeOpType = firstAndLastNodes(1).op_type;
                lastNodeOpType = firstAndLastNodes(2).op_type;
                nameStem = firstNodeOpType + "_To_" + lastNodeOpType;
            end
        end

        % Questions about the custom layers
        function onnxParams = onnxParamsForLayerName(this, layerName)
            [~,loc] = ismember(layerName, this.CustomLayerNames);
            onnxParams = this.ModelTranslations(loc).Params;
        end

        function names = customLayerInputTensorNames(this, layerName)
            [~,loc] = ismember(layerName, this.CustomLayerNames);
            names = string(this.ModelTranslations(loc).ToplevelGraphTranslation.ExternalInputNames);
        end

        function names = customLayerInputNumDimsNames(this, layerName)
            inputTensorNames = customLayerInputTensorNames(this, layerName);
            hasNumDims = isTensorWithNumDims(this, inputTensorNames);
            if any(hasNumDims)
                names = inputTensorNames(hasNumDims) + "NumDims";
            else
                names = string.empty;
            end
        end

        function names = customLayerOutputTensorNames(this, layerName)
            % Output tensor names, not including NumDims outputs.
            [~,loc] = ismember(layerName, this.CustomLayerNames);
            names = string(this.ModelTranslations(loc).ToplevelGraphTranslation.ExternalOutputNames);
        end

        function names = customLayerOutputNumDimsNames(this, layerName)
            outputTensorNames = customLayerOutputTensorNames(this, layerName);
            hasNumDims = isTensorWithNumDims(this, outputTensorNames);
            if any(hasNumDims)
                names = outputTensorNames(hasNumDims) + "NumDims";
            else
                names = string.empty;
            end
        end

        function tf = isNodeLastInSubgraph(this, nodeName, customLayerName)
            % Returns true if the node is (topologically) last in its
            % subgraph. This means all of the subgraph's input tensors are
            % available.
            [~,customLayerIdx] = ismember(customLayerName, this.CustomLayerNames);
            subgraphModelProto = this.ModelProtos(customLayerIdx);
            lastSubgraphNodeName = subgraphModelProto.graph.node(end).name;
            tf = isequal(nodeName, lastSubgraphNodeName);
        end

        function customLayerName = nodeCustomLayerName(this, nodeName)
            % Return the name of the auto-generated custom layer that
            % includes node nodeName, or empty if the node is not in a
            % custom layer.
            if isKey(this.NodeToCustomLayerMap, nodeName)
                customLayerName = this.NodeToCustomLayerMap(nodeName);
            else
                customLayerName = string.empty;
            end
        end

        function tf = isTensorLabeled(this, tensorName)
            tf = tensorLabel(this, tensorName) ~= "";
        end

        function str = tensorLabel(this, tensorName)
            str = tensorLabel(this.OnnxTensorLabeler, tensorName);
        end

        function tf = isTensorWithSize(this, tensorName)
            tf = ~isequal(this.TopLevelGraphProtoManager.variableSize(tensorName), -1);
        end

        function tfs = isTensorWithNumDims(this, tensorNames)
            % true for each tensor that needs an accompanying numdims
            % variable. This is all tensors that are unlabeled and which
            % are not output tensors of the top-level graph. It includes
            % unlabeled input tensors and internal tensors created by
            % auto-generated custom layers.
            isLabeled           = arrayfun(@(name)isTensorLabeled(this, name), tensorNames);
            isExternalOutput    = arrayfun(@(name)ismember(name, this.TopLevelGraphProtoManager.OutputNames), tensorNames);
            isWithSize          = arrayfun(@(name)isTensorWithSize(this, name), tensorNames);
            tfs                 = ~isLabeled & ~isExternalOutput & ~isWithSize;
        end
    end

    methods(Access=protected)
        function writeClassdefFile(this, className, code, isCoderClass)
            % className string
            % code char
            outputFilename = className + ".m";
            folderName = "+" + this.PackageName;
            if isCoderClass
                folderName = folderName + filesep + "+coder";
            end
            outputPath = fullfile(folderName, outputFilename);
            % Save code to file
            if isempty(dir(folderName))   % Search the current directory, not the path.
                try
                    mkdir(folderName);
                catch me
                    error(message('nnet_cnn_onnx:onnx:MkdirFailed', folderName, me.message));
                end
            end
            if ~isempty(dir(outputPath))   % Search the current directory, not the path.
                try
                    delete(outputPath);
                catch me
                    error(message('nnet_cnn_onnx:onnx:FileOverwrite', outputPath, me.message));
                end
            end
            [f, errmsg] = fopen(outputPath, 'w');
            if f < 0
                error(message('nnet_cnn_onnx:onnx:FileCreation', outputPath, errmsg));
            end
            fprintf(f, '%s', code);
            fclose(f);
        end

        function writeFunctions(this, coreFunctions, isCoderFunction)
            % folderName = "+" + this.PackageName;
            if isCoderFunction
                opsPackage = ['+' this.PackageName filesep '+coder' filesep '+ops'];
                opsPackageCoderCall = [this.PackageName '.coder.ops.'];
                utilityFcns = ["permuteInputVar", "permuteOutputVar", "extractIfDlarray"];
            else
                opsPackage = ['+' this.PackageName filesep '+ops'];
                opsPackageCall = [this.PackageName '.ops.'];
                utilityFcns = ["permuteInputVar", "permuteOutputVar"];
            end
            if ~exist(opsPackage, 'dir')
                mkdir(opsPackage);
            end
            coreFcns = unique(coreFunctions);
            if isempty(coreFcns)
                fcns = utilityFcns;
            else
                fcns = [coreFcns utilityFcns];
            end
            for i = 1:numel(fcns)
                if isCoderFunction
                    fullPath = which("nnet.internal.cnn.onnx.fcn.coder.ops." + fcns(i));
                else
                    fullPath = which("nnet.internal.cnn.onnx.fcn." + fcns(i));
                end
                fileID = fopen(fullPath);
                if fileID == -1
                    error(message('nnet_cnn_onnx:onnx:FileOpen', fcns(i)));
                end
                % Write op file to folder
                opFcnStr = fread(fileID,'*char')';
				fclose(fileID);
                if isCoderFunction
                    opFcnStr = strrep(opFcnStr, 'nnet.internal.cnn.onnx.fcn.coder.ops.', opsPackageCoderCall);
                else
                    opFcnStr = strrep(opFcnStr, 'nnet.internal.cnn.onnx.fcn.', opsPackageCall);
                end
                opFileName = strcat(fcns(i), ".m");
                [fid,~] = fopen(fullfile(opsPackage,opFileName), 'w+');
                fwrite(fid, opFcnStr);
                fclose(fid);
            end

            % Handle unsupported ops only in case of MATLAB
            unsupportedOps = getUnsupportedOps(this);
            if ~isempty(unsupportedOps)                         
                for j = 1:numel(unsupportedOps)
                    msg = message('nnet_cnn_onnx:onnx:PlaceholderFunctionErrorMsg', unsupportedOps{j});
                    if isCoderFunction
                        code = {"function varargout = onnx"+unsupportedOps{j}+"(varargin)"                         
                                "% Placeholder function implementation for ONNX operator: "+unsupportedOps{j}+""
                                "%#codegen"
                                "nnet.internal.cnn.coder.onnx.codererror(true, 'nnet_cnn_onnx:onnx:UnsupportedOperatorForCodegen','"+unsupportedOps{j}+"');"
                                "error("+msg.getString+");"
                                "end"};
                    else
                        code = {"function varargout = onnx"+unsupportedOps{j}+"(varargin)"                         
                                "% Placeholder function implementation for ONNX operator: "+unsupportedOps{j}+""
                                "error("+msg.getString+");"
                                "end"};
                    end
                    % Write unsupported op template file to folder
                    opFileName = strcat("onnx", unsupportedOps{j}, ".m");
                    [fid,~] = fopen(fullfile(opsPackage,opFileName), 'w+');
                    for k = 1 : length(code)
                        fprintf(fid, '%s\n', code{k});                        
                    end 
                    fclose(fid);
                end
                
            end
        end
        
        function unsupportedOps = getUnsupportedOps(this)
            translationIssues = getCustomLayerTranslationIssues(this);
            unsupportedOps = {};
            for i = 1:numel(translationIssues)
                if isa(translationIssues(i), 'nnet.internal.cnn.onnx.NodeTranslationError')
                    unsupportedOps{end+1} = char(translationIssues(i).Operator);
                end
            end
            unsupportedOps = unique(unsupportedOps);
        end

        function addToCustomLayerMap(this, nodeNames, customLayerName)
            for i=1:numel(nodeNames)
                this.NodeToCustomLayerMap(nodeNames(i)) = customLayerName;
            end
        end

        %% Segmenting out unsupported subgraphs, and generating custom layer code

        function digraphCell = segmentUnsupportedSubgraphs(this)
            % Toposort the graph, then rename the nodes by number. Then,
            % any subset of nodes can be toposorted by just sorting their
            % names numerically.
            [~, fullGraph] = toposort(this.NodeDigraph, 'Order', 'stable');
            origNodeNames = string(fullGraph.Nodes{:,:});                 	% Maps nodeNum-->original name
            nodeNums = 1:numel(origNodeNames);
            nodeNumMap = containers.Map;
            if ~isempty(origNodeNames)
                nodeNumMap = containers.Map(origNodeNames, nodeNums);          	% Maps origNodeName-->nodeNum
            end
            fullGraph.Nodes.Name = string(nodeNums)';                       % This renames the edges too
            % Make reachability and adjacency matrices
            FullReachabilityMatrix = iMakeReachabilityMatrix(fullGraph);
            FullAdjacencyMatrix = full(adjacency(fullGraph));
            % Delete fully-supported nodes
            FullySupportedNodeNums = arrayfun(@(name)nodeNumMap(name), this.FullySupportedNodeNames);
            unsupportedSubgraph = rmnode(fullGraph, FullySupportedNodeNums);
            % Check whether the subgraph is empty
            digraphCell = {};
            if numel(unsupportedSubgraph.Nodes) > 0
                % Make digraphs of unsupported nodes
                if ismember(this.CustomLayerContent, ["groupedOperators", "entireNetwork"])
                    % Find connected components covering all unsupported nodes
                    digraphCell = iConnectedComponentDigraphs(unsupportedSubgraph);
                else
                    assert(isequal(this.CustomLayerContent, "individualOperators"))
                    % Make each unsupported node into its own (trivial) digraph
                    digraphCell = iSingleNodeDigraphs(unsupportedSubgraph);
                end                    
                numComponents = numel(digraphCell);
                % Split concave digraphs into convex ones
                digraphCell = splitConcaveDigraphs(this, digraphCell, fullGraph, FullReachabilityMatrix, FullAdjacencyMatrix);
                % Restore the original names in all the graphs
                digraphCell = restoreOriginalSubgraphNames(this, digraphCell, origNodeNames);
                % Record whether any components were split, thereby
                % creating "coupled" custom layers
                if numel(digraphCell) > numComponents
                    this.AnyCoupledCustomLayers = true;
                end
            end
        end

        function digraphCell = splitConcaveDigraphs(this, digraphCell, fullGraph, FullReachabilityMatrix, FullAdjacencyMatrix)
            % Split convex digraphs into concave ones, delete them from the
            % list, and add the new ones to the end. Iterate in reverse so
            % we can stably modify the list as we go
            for i = numel(digraphCell):-1:1
                [isConcave, badOutputNodeNames] = isConcaveSubgraph(this, digraphCell{i}.Nodes.Name, fullGraph, FullReachabilityMatrix, FullAdjacencyMatrix);
                if isConcave
                    newDigraphs = splitOneConcaveDigraph(this, fullGraph, digraphCell{i}, badOutputNodeNames);
                    digraphCell(i) = [];
                    digraphCell = [digraphCell, newDigraphs]; %#ok<*AGROW>
                end
            end
        end

        function digraphCell = splitOneConcaveDigraph(~, fullGraph, dg, badOutputNodeNames)
            % Split the node list at the "bad" output nodes. First, sort
            % the nodes "globally topologically" so that all nodes upstream
            % of a 'bad' node precede it in the list. This is why we
            % renamed all the nodes as numbers, so we can do this on a
            % subgraph using 'sort'. It would NOT be enough to just sort
            % the subgraph topologically in isolation.
            digraphCell = {};
            dgNodeNames = string(sort(str2double(dg.Nodes.Name)));              % Globally toposort the dg node names.
            badOutputNodeNames = string(sort(str2double(badOutputNodeNames)));  % Globally toposort the badOutputNodeNames.
            % Split into subgraphs at each bad node
            [~, badLocs] = ismember(badOutputNodeNames, dgNodeNames);
            splitLocs = [0, badLocs(:)', numel(dgNodeNames)];                   % Note that the last node can't be bad
            for i = 2:numel(splitLocs)
                subgraphNodeNames = dgNodeNames(splitLocs(i-1)+1 : splitLocs(i));
                newDg = subgraph(fullGraph, subgraphNodeNames);
                % newDg may consist of disjoint components. Split them and
                % append them to the list
                newDgCell = iConnectedComponentDigraphs(newDg);
                digraphCell = [digraphCell, newDgCell];
            end
        end

        function [tf, badOutputNodeNames] = isConcaveSubgraph(~, nodeNames, fullGraph, FullReachabilityMatrix, FullAdjacencyMatrix)
            % Return true if the subgraph defined by nodeNames is a concave
            % subgraph of the full graph. Algorithm: Find all nodes that
            % are forward-reachable from any of the subgraph's nodes and
            % also backward-reachable from any of the subgraph's nodes.
            % (Note that this does not contain all nodes in the subgraph
            % itself.) If any such nodes are outside the subgraph, return
            % true. badOutputNodeNames is a cellstr of nodes in the
            % subgraph that send connections directly to external nodes.
            [~,nodeIdx] = ismember(nodeNames, fullGraph.Nodes.Name);
            fwd         = sum(FullReachabilityMatrix(nodeIdx,:), 1) > 0;
            bwd         = sum(FullReachabilityMatrix(:,nodeIdx), 2)' > 0;
            bothIdx     = find(fwd & bwd);
            externalIdx = setdiff(bothIdx, nodeIdx);
            tf          = ~isempty(externalIdx);
            % Find bad output nodes
            badOutputNodeNames = {};
            if tf
                block               = FullAdjacencyMatrix(nodeIdx, externalIdx);
                blockIdx            = sum(block, 2) > 0;  % Indices into the nodeIdx list
                badOutputIdx        = nodeIdx(blockIdx);
                badOutputNodeNames  = fullGraph.Nodes.Name(badOutputIdx);
            end
        end

        function digraphCell = restoreOriginalSubgraphNames(~, digraphCell, origNodeNames)
            % Restore the original names in all the graphs
            for i=1:numel(digraphCell)
                nodeNums = str2double(digraphCell{i}.Nodes.Name);
                origNames = origNodeNames(nodeNums);
                digraphCell{i}.Nodes.Name = origNames;                                  % This renames the edges too
            end
        end

        function modelProto = subgraphToModelProto(this, subgraphNodeNames, modelName, ...
                myDynamicTensorReads, allDynamicTensorReads)
            % Create a modelProto from a subset of nodes in a graph. The
            % nodes are assumed to constitute a connected component (DAG).
            % This modelProto will later be converted into a custom layer.
            [subgraphInputActivationNames, subgraphOutputActivationNames] = findSubgraphInputAndOutputNames(this, subgraphNodeNames, myDynamicTensorReads, allDynamicTensorReads);
            [inputVIPs, outputVIPs] = makeSubgraphInputAndOutputVIPs(this, subgraphInputActivationNames, subgraphOutputActivationNames);
            % Select the initializers used in this subgraph
            subgraphNodeProtos       = nodesFromNames(this.TopLevelGraphProtoManager, subgraphNodeNames);
            % Reorder node protos to match the topological order of
            % subgraphNodeNames (from the digraph), since nodesFromNames
            % returns them in file order which may not be topological.
            allNames = string({subgraphNodeProtos.name});
            [~, reorderIdx] = ismember(subgraphNodeNames, allNames);
            subgraphNodeProtos = subgraphNodeProtos(reorderIdx);
            allSubgraphInputNames    = iFindAllSubgraphInputNames(subgraphNodeProtos);
            subgraphInitializerNames = intersect(this.TopLevelGraphProtoManager.AllInitializerNames, allSubgraphInputNames);
            fullInitializers         = findFullInitializers(this.TopLevelGraphProtoManager, subgraphInitializerNames);
            sparseInitializers       = findSparseInitializers(this.TopLevelGraphProtoManager, subgraphInitializerNames);
            % First create a graphProto:
            %            node: [1×66 nnet.internal.cnn.onnx.NodeProto]
            %            name: 'squeezenet'
            %     initializer: [1×52 nnet.internal.cnn.onnx.TensorProto]
            %      doc_string: []
            %           input: [1×53 nnet.internal.cnn.onnx.ValueInfoProto]
            %          output: [1×1 nnet.internal.cnn.onnx.ValueInfoProto]
            %      value_info: []
            subgraphProto                    = nnet.internal.cnn.onnx.GraphProto;
            subgraphProto.name               = char(modelName);
            subgraphProto.node               = subgraphNodeProtos;
            subgraphProto.initializer        = fullInitializers;
            subgraphProto.sparse_initializer = sparseInitializers;
            subgraphProto.input              = inputVIPs;
            subgraphProto.output             = outputVIPs;
            % Now create the modelProto
            opsetIdProto = nnet.internal.cnn.onnx.OperatorSetIdProto;
            opsetIdProto.version        = int64(this.OpsetVersion);
            modelProto                  = nnet.internal.cnn.onnx.ModelProto;
            modelProto.ir_version       = int64(1.6);
            modelProto.opset_import     = opsetIdProto;
            modelProto.producer_name    = '';
            modelProto.producer_version = '';
            modelProto.domain           = '';
            modelProto.model_version  	= int64(1);
            modelProto.doc_string       = char(modelName);               % Storing the model name in it's doc_string field.
            modelProto.graph            = subgraphProto;
            modelProto.metadata_props	= [];
        end

        function [inputNames, outputNames] = findSubgraphInputAndOutputNames(this, subgraphNodeNames, myDynamicTensorReads, allDynamicReferencesCell)
            subgraphNodeProtos = nodesFromNames(this.TopLevelGraphProtoManager, subgraphNodeNames);
            allSubgraphOutputNames = unique([subgraphNodeProtos.output], 'stable');                               % Includes intermediate tensors
            allSubgraphOutputNames(cellfun(@isempty, allSubgraphOutputNames)) = [];
            nonSubgraphNodeNames = setdiff(this.TopLevelGraphProtoManager.NodeNames, subgraphNodeNames, 'stable');
            nonSubgraphNodeProtos = nodesFromNames(this.TopLevelGraphProtoManager, nonSubgraphNodeNames);
            % Find the activation inputs to the subgraph. These are the
            % node inputs minus tensors created inside the subgraph, minus
            % initializers:
            allInitializerNames = this.TopLevelGraphProtoManager.AllInitializerNames;
            inputNames = setdiff(unique([subgraphNodeProtos.input], 'stable'), allSubgraphOutputNames, 'stable');   % External inputs only (activations and weights).
            inputNames = setdiff(inputNames, allInitializerNames, 'stable');                      % External activation inputs only.
            inputNames(cellfun(@isempty, inputNames)) = [];
            % Add tensors that are created outside this custom layer and
            % read from my nested graphs via dynamic scoping
            inputNames = [inputNames, myDynamicTensorReads];
            % outputNames are all subgraph node outputs that are consumed
            % outside the subgraph. This includes (1) tensors that are
            % inputs to nonsubgraph nodes, (2) tensors that are external
            % outputs of the parent graph, and (3) tensors that are known
            % to be accessed by other layers' nested graphs but not created
            % by those layers:
            allConsumedTensorNames = unique([...
                this.TopLevelGraphProtoManager.OutputNames, ...    % Put outputs first so they're in the same order as in the network...
                [nonSubgraphNodeProtos.input],...
                [allDynamicReferencesCell{:}]...   % These are consumed by other layers but not necessarily created here.
                ], 'stable');
            outputNames = intersect(allConsumedTensorNames, allSubgraphOutputNames, 'stable');   % ... put outputs first here too.
        end

        function [inputVIPs, outputVIPs] = makeSubgraphInputAndOutputVIPs(this, subgraphInputActivationNames, subgraphOutputActivationNames)
            parentGraphMgr = this.TopLevelGraphProtoManager;
            % This subgraph may have input activation tensors that were not
            % inputs to the original graph, namely intermediate activation
            % tensors of the original graph. These intermedite tensors need
            % to have valueInfoProtos created for them.
            %   * First, copy known VIPs from full graph:
            [inputVIPFound, inputLocs] = ismember(subgraphInputActivationNames, parentGraphMgr.InputNames);
            inputVIPs(inputVIPFound) = parentGraphMgr.InputVIPs(inputLocs(inputVIPFound));
            %   * Next, make VIPs for intermediate tensors
            notFoundNames = subgraphInputActivationNames(~inputVIPFound);
            notFoundLocs = find(~inputVIPFound);
            for i=1:numel(notFoundNames)
                name = notFoundNames{i};
                loc = notFoundLocs(i);
                if isTensorLabeled(this, name)
                    % This input tensor is labeled, so create a VIP of known rank.
                    ndims = strlength(tensorLabel(this, name));
                    inputVIPs(loc) = nnet.internal.cnn.onnx.makeValueInfoProtoOfVariableSize(name, nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT, ndims);
                else
                    % This input tensor is not labeled, so create a VIP of unknown rank.
                    inputVIPs(loc) = nnet.internal.cnn.onnx.makeValueInfoProtoOfUnknownRank(name, nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT);
                end
            end
            % Special case of a custom layer with no inputs: Have it take
            % input from the first input to the network.
            if isempty(inputVIPs)
                dummyInputName = parentGraphMgr.InputNames{1};
                inputVIPs = nnet.internal.cnn.onnx.makeValueInfoProtoOfUnknownRank(dummyInputName, nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT);
            end
            % This subgraph may have output activation tensors that were
            % not outputs of the original graph, namely intermediate
            % activation tensors of the original graph. These intermediate
            % tensors need to have valueInfoProtos created for them.
            %   * First, copy known output tensor VIPs from full graph:
            [outputVIPFound, outputLocs] = ismember(subgraphOutputActivationNames, parentGraphMgr.OutputNames);
            outputVIPs(outputVIPFound) = parentGraphMgr.OutputVIPs(outputLocs(outputVIPFound));
            %   * Next, make VIPs for subgraph output tensors
            notFoundNames = subgraphOutputActivationNames(~outputVIPFound);
            notFoundLocs = find(~outputVIPFound);
            for i=1:numel(notFoundNames)
                name = notFoundNames{i};
                loc = notFoundLocs(i);
                if isTensorLabeled(this, name)
                    % This tensor has a known ONNX format.
                    ndims = strlength(tensorLabel(this, name));
                    outputVIPs(loc) = nnet.internal.cnn.onnx.makeValueInfoProtoOfVariableSize(name, nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT, ndims);	% ndims is specified but not the sizes.
                else
                    % This tensor has an unknown ONNX format.
                    outputVIPs(loc) = nnet.internal.cnn.onnx.makeValueInfoProtoOfUnknownRank(name, nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT);          % The shape of this VIP is empty.
                end
            end
        end

        %% Code generation
        function modelTranslation = genCustomLayerFunction(this, modelProto, nameStem, isCoderClass)
            maxNameLength = 20;
            modelFcnPath = "";  % Current directory
            modelFcnName = nameStem + "Fcn";
            modelTranslation = nnet.internal.cnn.onnx.fcn.ModelTranslation(modelProto, modelFcnPath, modelFcnName, maxNameLength, this.PackageName, true, isCoderClass);
        end

        function [className, DLTInputFormats, DLTOutputFormats, classFileCode] = genCustomLayerClass(this, modelTranslation, nameStem, generateCoderFiles, isCoderClass)
            % Generate custom layer class code and save to file. Model
            % function code is included in the file.
            className       = nnet.internal.cnn.onnx.fcn.uniqueLayerName(nameStem + "Layer");
            modelFcnName    = modelTranslation.ToplevelGraphTranslation.GraphName;
            learnableNames  = string(fieldnames(modelTranslation.Params.Learnables));
            stateNames = string(fieldnames(modelTranslation.Params.State));
            predictInputTensorNames = string(modelTranslation.ToplevelGraphTranslation.ExternalInputNames);
            needsNumDims = isTensorWithNumDims(this, predictInputTensorNames);
            if any(needsNumDims)
                predictInputNumDimsNames = predictInputTensorNames(needsNumDims) + "NumDims";
            else
                predictInputNumDimsNames = string.empty;
            end
            predictInputNames = [predictInputTensorNames predictInputNumDimsNames];
            inputModelFcnNumDimsNames = predictInputTensorNames + "NumDims";                 % Pass all input variable numdims to the modelfcn.
            numDimsVarsCode = genNumDimsVarsCode(this, predictInputTensorNames, isCoderClass);
            % Outputs
            predictOutputTensorNames  = string(modelTranslation.ToplevelGraphTranslation.ExternalOutputNames);
            needsNumDims = isTensorWithNumDims(this, predictOutputTensorNames);
            if any(needsNumDims)
                predictOutputNumDimsNames = predictOutputTensorNames(needsNumDims) + "NumDims";
            else
                predictOutputNumDimsNames = string.empty;
            end
            predictOutputNames = [predictOutputTensorNames predictOutputNumDimsNames];
            outputModelFcnNumDimsNames = predictOutputTensorNames + "NumDims";
            % Set DLTInputFormats. NumDims vars will have "".
            DLTInputFormats = repmat("", 1, numel(predictInputNames));
            for i=1:numel(predictInputTensorNames)
                tensorName = predictInputTensorNames(i);
                label = tensorLabel(this, tensorName);
                DLTInputFormats(i) = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.onnxLabelToDLTLabel(label);
            end
            % Set DLTOutputFormats. NumDims vars will have "".
            DLTOutputFormats = repmat("", 1, numel(predictOutputNames));
            for i=1:numel(predictOutputTensorNames)
                tensorName = predictOutputTensorNames(i);
                label = tensorLabel(this, tensorName);
                DLTOutputFormats(i) = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.onnxLabelToDLTLabel(label);
            end
            % Generate numInputs, numOutputs lines
            NumInputsLine = string.empty;
            if numel(predictInputNames)>1
                NumInputsLine = "this.NumInputs = " + numel(predictInputNames) + ";";
            end
            NumOutputsLine = string.empty;
            if numel(predictOutputNames)>1
                NumOutputsLine = "this.NumOutputs = " + numel(predictOutputNames) + ";";
            end
            % Generate OutputNames line
            OutputNamesLine = "this.OutputNames = {'" + strjoin(predictOutputNames, "', '") + "'};";
            % Code to apply dlarray formats to the layer's outputs            
            inputPermutationCode = genPermuteInput(this, modelTranslation.ToplevelGraphTranslation.ExternalInputShapes, predictInputTensorNames, isCoderClass);
            outputPermutationCode = genPermuteOutput(this, modelTranslation.ToplevelGraphTranslation.ExternalOutputShapes,predictOutputTensorNames, isCoderClass);
            outputFormatCode = genCodeToLabelOutputs(this, predictOutputTensorNames, predictOutputNumDimsNames, isCoderClass);
            % Assemble the classdef code
            classdefCode = assembleClassDefCode(className,learnableNames,stateNames,NumInputsLine,...
                                            NumOutputsLine,OutputNamesLine,predictOutputNames,...
                                            predictInputNames,predictInputTensorNames,numDimsVarsCode,inputPermutationCode,...
                                            predictOutputTensorNames, outputModelFcnNumDimsNames, modelFcnName, inputModelFcnNumDimsNames,...
                                            outputPermutationCode, outputFormatCode, modelTranslation.ModelFunctionCode, this.PackageName, generateCoderFiles, isCoderClass);            
            classFileCode = string(nnet.internal.cnn.onnx.util.indentcode(char(join(classdefCode',newline))));
        end        
        function code = genPermuteInput(this, inputSize, predictInputTensorNames, isCoderClass)
            % Generate code that will permute the input(s) when the user passes
            % permutation vectors to the model function via the
            % 'InputDataPermutation' NVP
            code = [];
            inPermCell = modelFcnInputDataPermutations(this, predictInputTensorNames);
            if isCoderClass
                fullyQualifiedCall = [this.PackageName '.coder.ops.permuteInputVar'];
                for i=1:numel(predictInputTensorNames)
                        code = [code sprintf('%s = %s(%s_, %s, %d);\n', ...
                                predictInputTensorNames{i}, fullyQualifiedCall, predictInputTensorNames{i},...
                                "[" + join(string(inPermCell{i}), ' ') + "]", numel(inputSize{i}))];
                end
            else
                fullyQualifiedCall = [this.PackageName '.ops.permuteInputVar'];
                for i=1:numel(predictInputTensorNames)
                    code = [code sprintf('%s = %s(%s, %s, %d);\n', ...
                            predictInputTensorNames{i}, fullyQualifiedCall, predictInputTensorNames{i},...
                            "[" + join(string(inPermCell{i}), ' ') + "]", numel(inputSize{i}))];
                end
            end
        end
                
        function code = genPermuteOutput(this, outputSize, predictOutputTensorNames, isCoderClass)
            code = [];
            outPermCell = modelFcnOutputDataPermutations(this, predictOutputTensorNames);
            if isCoderClass
                fullyQualifiedCall = [this.PackageName '.coder.ops.permuteOutputVar'];
                for i=1:numel(predictOutputTensorNames)
                    code = [code sprintf('%s_ = %s(%s__, %s, %d);\n', ...
                            predictOutputTensorNames{i}, fullyQualifiedCall, predictOutputTensorNames{i},...
                            "[" + join(string(outPermCell{i}), ' ') + "]", numel(outputSize{i}))];
                end
            else
                fullyQualifiedCall = [this.PackageName '.ops.permuteOutputVar'];
                for i=1:numel(predictOutputTensorNames)
                    code = [code sprintf('%s = %s(%s, %s, %d);\n', ...
                            predictOutputTensorNames{i}, fullyQualifiedCall, predictOutputTensorNames{i},...
                            "[" + join(string(outPermCell{i}), ' ') + "]", numel(outputSize{i}))];
                end
            end            
        end


        function permCell = modelFcnInputDataPermutations(this, inputTensorNames)
            % Find input permutation vectors. Labeled tensors need to be
            % permuted from DLT to reverse-onnx. Unlabeled tensors are
            % already in reverse-onnx.
            for i=1:numel(inputTensorNames)
                tensorName = inputTensorNames(i);
                if isTensorLabeled(this, tensorName)
                    permCell{i} = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.permDLTToONNX(tensorLabel(this, tensorName));
                else
                    % Tensor is already in reverse-ONNX dimension ordering
                    permCell{i} = "'as-is'";
                end
            end
        end
        
        function permCell = modelFcnOutputDataPermutations(this, outputTensorNames)
            % Find output permutation vectors. Labeled tensors need to be
            % permuted from reverse-onnx to DLT. Unlabeled tensors should
            % remain in reverse-onnx.
            for i=1:numel(outputTensorNames)
                tensorName = outputTensorNames(i);
                if isTensorLabeled(this, tensorName)
                    permCell{i} = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.permONNXToDLT(tensorLabel(this, tensorName));
                else
                    % Tensor stays in reverse-ONNX dimension ordering.
                    permCell{i} = "'as-is'";
                end
            end
        end

        function outputFormatCode = genCodeToLabelOutputs(this, predictOutputTensorNames, predictOutputNumDimsNames, isCoderClass)
            % Generate code to label the dlarray outputs of a custom
            % layer's predict method
            outputFormatCode = string.empty;
            outputLabelCell = arrayfun(@(name)tensorLabel(this, name), predictOutputTensorNames, 'UniformOutput',false);            
            if isCoderClass
                % Label the output tensors
                for i=1:numel(predictOutputTensorNames)
                    if outputLabelCell{i}==""
                        % The output tensor has no label. Generate code that
                        % formats it with U's at runtime
                        outputNumDimsName = predictOutputTensorNames(i) + "NumDims";
                        fmt = "repmat('U', 1, max(2, coder.const(" + outputNumDimsName + "__)))";
                        outputFormatCode(i,1) = predictOutputTensorNames(i) + " = dlarray(single(" + predictOutputTensorNames(i) + "_), " + fmt + ");";
                    else
                        % The output tensor has a label.
                        fmt = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.onnxLabelToDLTLabel(outputLabelCell{i});
                        outputFormatCode(i,1) = predictOutputTensorNames(i) + " = dlarray(single(" + predictOutputTensorNames(i) + "_), '" + fmt + "');";
                    end
                end

                % Now construct and label the output numDims vars. To pass
                % NumDims between layers, we encode it as a vector of ones,
                % with 'UU' dlarray format. To make sure it respects GPU/CPU
                % status, we make its type 'like' that of the layer's first
                % output tensor.
                for j=1:numel(predictOutputNumDimsNames)
                    % Generate code that formats it with UU at runtime
                    outputFormatCode(numel(predictOutputTensorNames)+j,1) = ...
                        predictOutputNumDimsNames(j) + " = dlarray(ones(1," + predictOutputNumDimsNames(j) + "__,'like'," + predictOutputTensorNames(1) + "), 'UU');";
                end
            else
                % Label the output tensors
                for i=1:numel(predictOutputTensorNames)
                    if outputLabelCell{i}==""
                        % The output tensor has no label. Generate code that
                        % formats it with U's at runtime
                        outputNumDimsName = predictOutputTensorNames(i) + "NumDims";
                        fmt = "repmat('U', 1, max(2, " + outputNumDimsName + "))";
                        outputFormatCode(i,1) = predictOutputTensorNames(i) + " = dlarray(single(" + predictOutputTensorNames(i) + "), " + fmt + ");";
                    else
                        % The output tensor has a label.
                        fmt = nnet.internal.cnn.onnx.ModelTranslationIntoLayers.onnxLabelToDLTLabel(outputLabelCell{i});
                        outputFormatCode(i,1) = predictOutputTensorNames(i) + " = dlarray(single(" + predictOutputTensorNames(i) + "), '" + fmt + "');";
                    end
                end

                % Now construct and label the output numDims vars. To pass
                % NumDims between layers, we encode it as a vector of ones,
                % with 'UU' dlarray format. To make sure it respects GPU/CPU
                % status, we make its type 'like' that of the layer's first
                % output tensor.
                for j=1:numel(predictOutputNumDimsNames)
                    % Generate code that formats it with UU at runtime
                    outputFormatCode(numel(predictOutputTensorNames)+j,1) = ...
                        predictOutputNumDimsNames(j) + " = dlarray(ones(1," + predictOutputNumDimsNames(j) + ",'like'," + predictOutputTensorNames(1) + "), 'UU');";
                end
            end            
        end

        function numDimsVarsCode = genNumDimsVarsCode(this, predictInputTensorNames, isCoderClass)
            % Gen code to assign values to numdims vars in this custom
            % layer
            numDimsVarsCode = string.empty;
            for i=1:numel(predictInputTensorNames)
                tensorSize = this.TopLevelGraphProtoManager.variableSize(predictInputTensorNames(i));
                tensorName = string(predictInputTensorNames(i));
                numDimsClassVarName = tensorName + "NumDims";
                numDimsVarName  = tensorName + "NumDims";
                label = tensorLabel(this, tensorName);
                if label=="" && isequal(tensorSize, -1)
                    % For unlabeled tensors, numDims is passed as an input
                    % to its predict() and forward() methods as a vector of
                    % length NumDims.
                    if isCoderClass
                        numDimsVarsCode = [numDimsVarsCode; numDimsClassVarName + " = numel(" + numDimsVarName + "__);"];
                    else
                        numDimsVarsCode = [numDimsVarsCode; numDimsClassVarName + " = numel(" + numDimsVarName + ");"];
                    end
                else
                    if label~= ""
                        % For labeled tensors, we can derive NumDims from the
                        % label
                        numDimString = string(strlength(label));
                    else
                        % For tensors with sizes, we can derive NumDims from
                        % sizes
                        numDimString = string(numel(tensorSize));
                    end
                    numDimsVarsCode = [numDimsVarsCode; numDimsClassVarName + " = " + numDimString + ";"];
                end
            end
        end
    end

    methods
        function issues = getCustomLayerTranslationIssues(this)
            % Get the set of translation issues across all generated custom
            % layers.
            issues = arrayfun(@(mt) getTranslationIssues(mt), this.ModelTranslations, 'UniformOutput', false);
            issues = [issues{:}];
        end
    end
end

%% File-internal helper functions
function code = assembleClassDefCode(className,learnableNames,stateNames,NumInputsLine,...
                                    NumOutputsLine,OutputNamesLine,predictOutputNames,...
                                    predictInputNames,predictInputTensorNames,numDimsVarsCode,inputPermutationCode,...
                                    predictOutputTensorNames, outputModelFcnNumDimsNames, modelFcnName, inputModelFcnNumDimsNames,...
                                    outputPermutationCode, outputFormatCode, modelFunctionCode, packageName, generateCoderFiles, isCoderClass)
        code = [
        "classdef " + className + " < nnet.layer.Layer & nnet.layer.Formattable"
        "% " + getString(message('nnet_cnn_onnx:onnx:CustomLayerDescription'))
        codegenPragmaBlock(isCoderClass)
        ""
        "%#ok<*PROPLC>"
        "%#ok<*NBRAK>"
        "%#ok<*INUSL>"
        "%#ok<*VARARG>"
        "properties (Learnable)"
        learnableNames
        "end"
        ""
        "properties (State)"
        statesBlock(stateNames, isCoderClass)
        
        "end"
        ""
        "properties"
        "Vars"
        "NumDims"
        "end"
        ""
        matlabCodegenNonTunablePropsBlock(learnableNames, stateNames, isCoderClass)
        matlabCodegenRedirectBlock(className, packageName, generateCoderFiles, isCoderClass)
        matlabCodegenRedirectedBlocks(className, packageName, learnableNames, stateNames, isCoderClass)
        "methods"
        constructorBlock(className, NumInputsLine, NumOutputsLine, OutputNamesLine, learnableNames, stateNames, isCoderClass, packageName)
        predictAndForwardBlock(predictOutputNames, stateNames, predictInputNames, predictInputTensorNames, ...
                                 numDimsVarsCode, inputPermutationCode, predictOutputTensorNames, ...
                                 outputModelFcnNumDimsNames, modelFcnName, inputModelFcnNumDimsNames, ...
                                 outputPermutationCode, outputFormatCode, isCoderClass)
        modelFunctionCode
        "end"
        ];
end

function cgPragma = codegenPragmaBlock(isCoderClass)
    if isCoderClass
        cgPragma = "%#codegen";
    else
        cgPragma = string.empty;
    end
end

function states = statesBlock(stateNames, isCoderClass)
    if isCoderClass
        states = string.empty;
    else
        states = stateNames;
    end
end

function predFwdBlk = predictAndForwardBlock(predictOutputNames, stateNames, predictInputNames, predictInputTensorNames, ...
                                             numDimsVarsCode, inputPermutationCode, predictOutputTensorNames, ...
                                             outputModelFcnNumDimsNames, modelFcnName, inputModelFcnNumDimsNames, ...
                                             outputPermutationCode, outputFormatCode, isCoderClass)
    if isCoderClass
        predictCode = coderPredictCode(predictOutputTensorNames, outputModelFcnNumDimsNames, stateNames, modelFcnName, predictInputTensorNames, inputModelFcnNumDimsNames);
        % Dont need a forward method for coder layer
        predFwdBlk = [
        "function " + "[" + join([predictOutputNames stateNames'], ", ") + "] = predict(this, " + join(predictInputNames, "__, ") + "__)"
        iStripInputDimsCode(predictInputTensorNames, isCoderClass)
        numDimsVarsCode
        inputPermutationCode
        predictCode
        outputPermutationCode
        outputFormatCode
        "end"
        ""        
        ];
    else
        predFwdBlk = [
        "function " + "[" + join([predictOutputNames stateNames'], ", ") + "] = predict(this, " + join(predictInputNames, ", ") + ")"
        iStripInputDimsCode(predictInputTensorNames, isCoderClass)
        numDimsVarsCode
        inputPermutationCode
        "[" + join([predictOutputTensorNames outputModelFcnNumDimsNames stateNames'], ", ") + "] = " + modelFcnName + "(this, " + join([predictInputTensorNames inputModelFcnNumDimsNames], ", ") + ", false);"
        outputPermutationCode
        outputFormatCode
        "end"
        ""
        "function " + "[" + join([predictOutputNames stateNames'], ", ") + "] = forward(this, " + join(predictInputNames, ", ") + ")"
        iStripInputDimsCode(predictInputTensorNames, isCoderClass)
        numDimsVarsCode
        inputPermutationCode
        "[" + join([predictOutputTensorNames outputModelFcnNumDimsNames stateNames'], ", ") + "] = " + modelFcnName + "(this, " + join([predictInputTensorNames inputModelFcnNumDimsNames], ", ") + ", true);"
        outputPermutationCode
        outputFormatCode
        "end"
        ""
        ];
    end    
end
function predictCode = coderPredictCode(predictOutputTensorNames, outputModelFcnNumDimsNames, stateNames, modelFcnName, predictInputTensorNames, inputModelFcnNumDimsNames)
if isempty(stateNames)
    predictCode = "[" + join([predictOutputTensorNames outputModelFcnNumDimsNames], "__, ") + "__] = " + modelFcnName + "(this, " + join([predictInputTensorNames inputModelFcnNumDimsNames], ", ") + ", false);";
else
    predictCode = "[" + join([predictOutputTensorNames outputModelFcnNumDimsNames], "__, ") + "__, " + join(stateNames', ", ") + "] = " + modelFcnName + "(this, " + join([predictInputTensorNames inputModelFcnNumDimsNames], ", ") + ", false);";
end

end
function cstrBlk = constructorBlock(className, NumInputsLine, NumOutputsLine, OutputNamesLine, learnableNames, stateNames, isCoderClass, packageName)    
    if isCoderClass
        cstrBlk = [
                "function this = " + className + "(mlInstance)"
                "this.Name = mlInstance.Name;"
                NumInputsLine
                NumOutputsLine
                OutputNamesLine
                genCoderVarsAssignmentCode(packageName)
                "this.NumDims = mlInstance.NumDims;"                            
                genVarAssignmentCode(learnableNames, "mlInstance", "this")
                genStateAssignmentCode(stateNames, "mlInstance", "this", 0)
                "end"
                ""
                ];
    else
        cstrBlk = [
                "function this = " + className + "(name)"
                "this.Name = name;"
                NumInputsLine
                NumOutputsLine
                OutputNamesLine     
                "end"
                ""
                ];
    end
end

function varsAssignmentCode = genCoderVarsAssignmentCode(packageName)
varsAssignmentCode = [
                        "if isstruct(mlInstance.Vars)"
                        "names = fieldnames(mlInstance.Vars);"
                        "for i=1:numel(names)"                    
                        "fieldname = names{i};"
                        "this.Vars.(fieldname) = " + packageName + ".coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));"               
                        "end"
                        "else"
                        "this.Vars = [];"
                        "end"
                        ""
                        ];
end

function varAssignmentCode = genVarAssignmentCode(vars, fromInstanceStr, toInstanceStr)
    if ~isempty(vars)
        varAssignmentCode = toInstanceStr + "." + vars + " = " + fromInstanceStr + "." + vars + ";";        
    else
       varAssignmentCode = string.empty;
    end
end

function stateAssignmentCode = genStateAssignmentCode(vars, fromInstanceStr, toInstanceStr, toMATLABFlag)
if ~isempty(vars) && toMATLABFlag
    stateAssignmentCode = toInstanceStr + "." + vars + " = " + fromInstanceStr + ".Vars." + vars + ";";
elseif ~isempty(vars) && ~toMATLABFlag
    stateAssignmentCode = toInstanceStr + ".Vars." + vars + " = " + fromInstanceStr + "." + vars + ";";
else
    stateAssignmentCode = string.empty;
end

end
function mlctfrb = matlabCodegenRedirectedBlocks(className, packageName, learnableNames, stateNames, isCoderClass)
    if isCoderClass
        mlctfrb = [
            "methods(Static, Hidden)"
            "% Instantiate a codegenable layer instance from a MATLAB layer instance"                    
            "function this_cg = matlabCodegenToRedirected(mlInstance)"
            "this_cg = " + packageName + ".coder." + className + "(mlInstance);" 
            "end"
            "function this_ml = matlabCodegenFromRedirected(cgInstance)"
            "this_ml = " + packageName + "." + className + "(cgInstance.Name);"
            "if isstruct(cgInstance.Vars)"
            "names = fieldnames(cgInstance.Vars);"
            "for i=1:numel(names)"                    
            "fieldname = names{i};"
            "this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));"               
            "end"
            "else"
            "this_ml.Vars = [];"
            "end"            
            "this_ml.NumDims = cgInstance.NumDims;"
            genVarAssignmentCode(learnableNames, "cgInstance", "this_ml")
            genStateAssignmentCode(stateNames, "cgInstance", "this_ml", 1)
            "end"
            "end"
            ""
            ];
    else
        mlctfrb = "";
    end
end

function mlcrb = matlabCodegenRedirectBlock(className, packageName, generateCoderFiles, isCoderClass)
    if generateCoderFiles && ~isCoderClass
        mlcrb = [
            "methods(Static, Hidden)"
            "% Specify the path to the class that will be used for codegen"                    
            "function name = matlabCodegenRedirect(~)"
            "name = '" + packageName + ".coder." + className + "';";
            "end"
            "end"
            ""
            ];
    else
        mlcrb = "";
    end
end

function mlcntpb = matlabCodegenNonTunablePropsBlock(learnableNames, stateNames, isCoderClass)
    if isCoderClass
        mlcntpb = [
            "methods(Static, Hidden)"
            "% Specify the properties of the class that will not be modified"
            "% after the first assignment."
            "function p = matlabCodegenNontunableProperties(~)"
            "p = {"
            "% Constants, i.e., Vars, NumDims and all learnables and states"
            "'Vars'"
            "'NumDims'"              
            "};"
            "end"
            "end"
            ""
            ];
    else
        mlcntpb = "";
    end
end

function quotedVarCode = genQuotedVarCode(vars)
    if ~isempty(vars)
       quotedVarCode = "'"+ join(vars, ['''' newline '''']) + "'";
    else
       quotedVarCode = string.empty;
    end
end

function digraphCell = iConnectedComponentDigraphs(dg)
% Split a digraph into its disjoint connected component digraphs.
digraphCell = {dg};
ccbins = conncomp(dg, 'Type','weak');                           % Find connected components. 'weak' means use connections in both directions.
% Make digraphs from the connected components
for i=1:max(ccbins)
    names               = dg.Nodes.Name(ccbins==i);
    digraphCell{i}      = subgraph(dg, names);
end
end

function digraphCell = iSingleNodeDigraphs(dg)
% Split a digraph into all single-node digraphs.
digraphCell = {dg};
% Make digraphs from the bins
for i=1:height(dg.Nodes)
    name               = dg.Nodes.Name(i);
    digraphCell{i}     = subgraph(dg, name);
end
end

function R = iMakeReachabilityMatrix(dg)
% R(i,j)>0 iff there is a directed path from i to j.
D = transclosure(dg);
R = full(adjacency(D));
end

function code = iStripInputDimsCode(inputTensorNames, isCoderClass)
if isCoderClass
    code = arrayfun(@(name)"if isdlarray(" + name + "__)" + newline + name + "_ = stripdims(" + name + "__);"...
                    + newline + "else" + newline + name + "_ = " + name + "__;" + newline + "end", inputTensorNames);
else
    code = arrayfun(@(name)"if isdlarray(" + name + ")" + newline + name + " = stripdims(" + name + ");" + newline + "end", inputTensorNames);
end
code = code(:);
end

function Names = iFindAllSubgraphInputNames(nodeProtos)
Names = iUniqueInputNamesInGraphTree(nodeProtos);
end

function Names = iUniqueInputNamesInGraphTree(nodeProtos)
% Find all the unique names in the graph tree.
% First, collect all names from the current flat graph:
if ~isempty(nodeProtos)
    Names = [nodeProtos.input];
    % Recursively add names from subgraphs:
    for i=1:numel(nodeProtos)
        Names = [Names iNamesFromNode(nodeProtos(i))];
    end
    % Remove empty names
    isEmptyName = cellfun(@isempty, Names);
    Names(isEmptyName) = [];
    % Get unique names
    Names = unique(Names, 'stable');
else
    Names = {};
end
end

function Names = iNamesFromNode(nodeProto)
% Recursively find all the names in subgraph trees under this node
Names = {};
for i = 1:numel(nodeProto.attribute)
    subgraphs = [nodeProto.attribute(i).g, nodeProto.attribute(i).graphs];
    for g = 1:numel(subgraphs)
        Names = [Names, iUniqueInputNamesInGraphTree(subgraphs(g).node)];
    end
end
end
