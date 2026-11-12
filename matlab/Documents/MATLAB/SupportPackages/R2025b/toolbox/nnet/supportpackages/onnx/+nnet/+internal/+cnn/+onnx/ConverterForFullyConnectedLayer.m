classdef ConverterForFullyConnectedLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a FullyConnectedLayer into ONNX
    
    % Copyright 2018-2021 The MathWorks, Inc.
    
    properties
        LayerAnalyzer
    end
    
    methods
        function this = ConverterForFullyConnectedLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
            this.LayerAnalyzer = layerAnalyzer;
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            existingNodeNames   = {nodeProto.name};
            onnxName            = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            switch inputTensorLayout
                case {'nchw', 'nchwd','nch'}
                    % Output a single Conv node
                    if isequal(inputTensorLayout,'nch')
                        imgDim      = 1;
                        weightPerm  = [3 2 1]; % NNT is HCF. ONNX is FCH.
                    elseif isequal(inputTensorLayout, 'nchw')
                        imgDim      = 2;
                        weightPerm  = [4 3 1 2];     % NNT W is HWCF. ONNX is FCHW.
                    else
                        imgDim      = 3;
                        weightPerm  = [5 4 1 2 3];   % NNT W is HWDCF. ONNX is FCHWD.
                    end
                    W                   = this.LayerAnalyzer.InternalLayer.Weights.Value;
                    onnxName            = makeUniqueName(existingNodeNames, onnxName);
                    WName               = [onnxName '_W'];
                    BName               = [onnxName '_B'];
                    convNode            = NodeProto;
                    convNode.op_type	= 'Conv';
                    convNode.name       = onnxName;
                    convNode.input      = mapTensorNames(this, {this.InputLayerNames{1}, WName, BName}, TensorNameMap);
                    convNode.output     = {onnxName};
                    nntWeightSize      	= size(W);                  % hcf or hwcf or hwdcf
                    kernel_shape        = nntWeightSize(1:imgDim);  % h or hw or hwd
                    convNode.attribute = [...
                        makeAttributeProto('group',        'INT',  1),...
                        makeAttributeProto('dilations',    'INTS', ones(1,imgDim)),...
                        makeAttributeProto('kernel_shape', 'INTS', kernel_shape),...
                        makeAttributeProto('pads',         'INTS', zeros(1,2*imgDim)),...
                        makeAttributeProto('strides',      'INTS', ones(1,imgDim))
                        ];
                    % Make parameter Initializers for: W, B
                    t1              = TensorProto;
                    t1.name         = WName;
                    t1.data_type	= TensorProto_DataType.FLOAT;
                    permutedW       = permute(W, weightPerm);                            
                    t1.raw_data     = rawData(single(permutedW));
                    t1.dims         = dimVector(size(permutedW), imgDim+2);
                    
                    t2              = TensorProto;
                    t2.name         = BName;
                    t2.data_type    = TensorProto_DataType.FLOAT;
                    t2.raw_data     = rawData(single(squeeze(this.NNTLayer.Bias)));     % NNT Bias is 11F or 111F
                    t2.dims         = dimVector(numel(this.NNTLayer.Bias),1);            
                    
                    nodeProto(end+1)        = convNode;
                    parameterInitializers   = [t1 t2];
                    networkInputs           = [];
                    networkOutputs          = [];
                    
                    % Update maps
                    TensorNameMap(this.NNTLayer.Name) = onnxName;
                    TensorLayoutMap(onnxName)         = inputTensorLayout;
                case {'snc', '1nc','nc'}
                    % Output MatMul-->Add
                    matMulNodeName      = [onnxName '_MatMul'];
                    addNodeName         = [onnxName '_Add'];
                    matMulNodeName      = makeUniqueName(existingNodeNames, matMulNodeName);
                    addNodeName         = makeUniqueName([existingNodeNames, {matMulNodeName}], addNodeName);
                    WName               = [matMulNodeName '_W'];
                    BName               = [addNodeName '_B'];
                    % MatMul
                    MatMulNode        	= NodeProto;
                    MatMulNode.op_type	= 'MatMul';
                    MatMulNode.name    	= matMulNodeName;
                    MatMulNode.input    = [inputTensorNames(1), {WName}];
                    MatMulNode.output   = {matMulNodeName};
                    
                    % Add
                    AddNode            = NodeProto;
                    AddNode.op_type    = 'Add';
                    AddNode.name       = addNodeName;
                    AddNode.input      = {matMulNodeName, BName};
                    AddNode.output     = {addNodeName};
                    
                    % Make parameter Initializer for WName.
                    % Weights are stored in the external layer as FC, where
                    % C is numHidden from the previous layer. ONNX needs CF,
                    % so first we transpose, then use rawData() to convert
                    % it to row-major.
                    W               = single(this.NNTLayer.Weights)';
                    t1              = TensorProto;
                    t1.name         = WName;
                    t1.data_type	= TensorProto_DataType.FLOAT;
                    t1.raw_data     = rawData(W);
                    t1.dims         = dimVector(size(W),2);
                    % Make parameter Initializer for BName:
                    t2              = TensorProto;
                    t2.name         = BName;
                    t2.data_type	= TensorProto_DataType.FLOAT;
                    t2.raw_data     = rawData(single(this.NNTLayer.Bias));
                    t2.dims         = dimVector(numel(this.NNTLayer.Bias),1);   % Bias is a 1D vector: size = F.
                    
                    nodeProto               = [nodeProto MatMulNode AddNode];
                    parameterInitializers   = [t1 t2];
                    networkInputs           = [];
                    networkOutputs          = [];
                    
                    % Update maps
                    outputTensorName                  = addNodeName;
                    TensorNameMap(this.NNTLayer.Name) = outputTensorName;
                    TensorLayoutMap(outputTensorName) = inputTensorLayout;
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
