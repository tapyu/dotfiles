classdef ConverterForNetwork
    % Class to convert a Network into ONNX

    %  Example ModelProto:
    %           ir_version: 1
    %         opset_import: []
    %        producer_name: 'onnx-caffe2'
    %     producer_version: []
    %               domain: []
    %        model_version: []
    %           doc_string: []
    %                graph: [1x1 nnet.internal.cnn.onnx.GraphProto]
    %       metadata_props: []

    % Copyright 2018-2025 The Mathworks, Inc.

    properties
        Network
        Metadata
        BatchSizeToExport   % An integer or a string
    end

    methods
        function this = ConverterForNetwork(network, metadata, BatchSizeToExport)
            this.Network = network;
            this.Metadata = metadata;
            this.BatchSizeToExport = BatchSizeToExport;
        end

        function modelProto = toOnnx(this)
            import nnet.internal.cnn.onnx.*
            % Set ONNX operatorSet version number
            opsetIdProto = OperatorSetIdProto;
            opsetIdProto.version = int64(this.Metadata.OpsetVersion);

            % ModelProto fields
            modelProto                  = ModelProto;
            modelProto.ir_version       = int64(this.Metadata.getIrVersion(opsetIdProto.version));
            modelProto.opset_import     = opsetIdProto;
            modelProto.producer_name    = char(this.Metadata.ProducerName);
            modelProto.producer_version = char(this.Metadata.ProducerVersion);
            modelProto.domain           = char(this.Metadata.Domain);
            modelProto.model_version  	= int64(this.Metadata.ModelVersion);
            modelProto.doc_string       = char(this.Metadata.DocString);
            modelProto.graph            = networkToGraphProto(this);
            modelProto.metadata_props	= [];

            % Optionally add MathWorks operatorSet
            if MathWorksOperatorsUsed(modelProto)
                opsetIdProto = OperatorSetIdProto;
                opsetIdProto.domain = 'com.mathworks';
                opsetIdProto.version = int64(this.Metadata.MathWorksOpsetVersion);
                modelProto.opset_import(end+1) = opsetIdProto;
            end
        end

        function graphProto = networkToGraphProto(this)
            %   Example GraphProto:
            %            node: [1×66 nnet.internal.cnn.onnx.NodeProto]
            %            name: 'squeezenet'
            %     initializer: [1×52 nnet.internal.cnn.onnx.TensorProto]
            %      doc_string: []
            %           input: [1×53 nnet.internal.cnn.onnx.ValueInfoProto]
            %          output: [1×1 nnet.internal.cnn.onnx.ValueInfoProto]
            %      value_info: []
            import nnet.internal.cnn.onnx.exporter.*

            % Convert layers while gathering nodes, initializers, network
            % inputs, network outputs, and the TensorNameMap.
            nodeProtos              = nnet.internal.cnn.onnx.NodeProto.empty;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];

            % The TensorNameMap maps a tensor name in the DLT network to
            % the name of the ONNX tensor that plays the same role in the
            % ONNX graph. For example: Suppose a DLT fullyConnectedLayer
            % has the name "FC1". We consider the name "FC1" to refer to
            % the layer's output tensor. When exported to ONNX, that layer
            % may expand into 2 operators, e.g., Flatten followed by Gemm.
            % The output tensor of the Gemm operator may be given the name
            % "FC1_Gemm". In that case, the association {"FC1", "FC1_Gemm"}
            % will be added to the TensorNameMap, so that
            % TensorNameMap('FC1') = 'FC1_Gemm'. All tensors whose names
            % change will be added to the TensorNameMap.
            TensorNameMap           = containers.Map;

            % The TensorLayoutMap maps an ONNX tensor name to a layout
            % string. Not all tensors are included. Examples:
            %   TensorLayoutMap('Conv_1',   'nchw')
            %   TensorLayoutMap('FC1_Gemm', 'nc')
            %   TensorLayoutMap('LSTM_1',   'snc').
            TensorLayoutMap         = containers.Map;
            networkAnalysis         = nnet.internal.cnn.analyzer.NetworkAnalyzer(this.Network);

            % Valid network for dlnetwork and layergraph
            % return names of dangling layers if no output layer for
            % dlnetwork and layergraph
            danglingLayerNames = iValidateNetwork(this, networkAnalysis);

            IsRecurrentNetwork      = isa(this.Network.Layers(1), 'nnet.cnn.layer.SequenceInputLayer');
            for layerNum = 1:numel(networkAnalysis.LayerAnalyzers)
                % LayerAnalyzers are in topological order
                layerAnalyzer           = networkAnalysis.LayerAnalyzers(layerNum);
                isDanglingLayer         = ismember(layerAnalyzer.Name, danglingLayerNames);
                layerConverter          = NNTLayerConverter.makeLayerConverter(layerAnalyzer, ...
                    this.Metadata.OpsetVersion, IsRecurrentNetwork, isDanglingLayer, this.BatchSizeToExport);
                if isa(layerConverter.NNTLayer,"nnet.cnn.layer.Resize2DLayer") || isa(layerConverter.NNTLayer,"nnet.cnn.layer.Resize3DLayer")
                    [nodeProtos, paramInitializers, netInputs, netOutputs, TensorNameMap, TensorLayoutMap] ...
                        = toOnnx(layerConverter, nodeProtos, TensorNameMap, TensorLayoutMap, networkAnalysis.Connections);
                else
                    [nodeProtos, paramInitializers, netInputs, netOutputs, TensorNameMap, TensorLayoutMap] ...
                        = toOnnx(layerConverter, nodeProtos, TensorNameMap, TensorLayoutMap);
                end
                parameterInitializers   = [parameterInitializers, paramInitializers];
                networkInputs           = [networkInputs, netInputs];
                networkOutputs          = [networkOutputs, netOutputs];
            end

            % Set graphProto fields
            graphProto              = nnet.internal.cnn.onnx.GraphProto;
            graphProto.name         = this.Metadata.NetworkName;
            graphProto.node         = nodeProtos;
            graphProto.initializer	= parameterInitializers;
            graphProto.input        = networkInputs;
            graphProto.output       = networkOutputs;
        end

        function danglingLayerNames = iValidateNetwork(this, networkAnalysis)
            % Validate network only for layergraph and dlnetwork
            danglingLayerNames = {};
            constraints = nnet.internal.cnn.onnx.exporter.getConverterConstraints();
            converterConstraintFilter = nnet.internal.cnn.onnx.constraints.ConverterConstraintFilter(constraints);
            networkAnalysis.applyConstraints(converterConstraintFilter);
            ids = networkAnalysis.Issues.Id;

            if ~isempty(ids)
                % If has any error other than the filtered issues,
                % throw error
                networkAnalysis.throwIssuesIfAny();
            end

            if ~converterConstraintFilter.HasMissingOutput
                % If there are no missing outputs, return
                return
            end
    
            % If current network is dlnetwork, get its layergraph for
            % extracting dangling nodes
            if isa(this.Network, 'dlnetwork')
                netLG = layerGraph(this.Network);
            else
                netLG = this.Network;
            end
            % Get dangling nodes for layergraph
            digraph = netLG.extractPrivateDirectedGraph();
            outputLayers = netLG.Layers(digraph.outdegree == 0);
            danglingLayerNames = {outputLayers.Name};
        end
    end
end

function tf = MathWorksOperatorsUsed(modelProto)
nodes = modelProto.graph.node;
tf = any(arrayfun(@(n)isequal(lower(n.domain), 'com.mathworks'), nodes));
end
