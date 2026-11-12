classdef ConverterForGroupedConvolution2DLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a grouped convolution2dLayer into ONNX

    % Copyright 2018-2024 The MathWorks, Inc.

    methods
        function this = ConverterForGroupedConvolution2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            convOperatorInput = this.InputLayerNames{1};
            permutedPadding   = this.NNTLayer.PaddingSize([1,3,2,4]);

            padInits     = [];
            paddingValue = this.NNTLayer.PaddingValue;
            if isequal(paddingValue,"symmetric-include-edge")
                error(message("nnet_cnn_onnx:onnx:ConvPaddingValueUnsupportedForExport", class(this.NNTLayer)));
            elseif paddingValue~=0
                padOnnxName     = makeUniqueName({nodeProto.name}, 'pad');
                padInput   = mapTensorNames(this, {this.InputLayerNames{1}}, TensorNameMap);
                padOutput  = {padOnnxName};

                if isequal(paddingValue,"symmetric-exclude-edge")
                    padOnnxMode = 'reflect';
                elseif isequal(paddingValue,"replicate")
                    padOnnxMode = 'edge';
                else
                    padOnnxMode = 'constant';
                end

                % The ONNX pad operator requires as many elements as twice
                % the number of dimensions. add zero-sized padding
                % corresponding to the batch and channel dimensions.
                beginPadding = [zeros(1,2) permutedPadding(1:2)];
                endPadding   = [zeros(1,2) permutedPadding(3:end)];
                pads         = [beginPadding endPadding];

                [padNode, padInits] = createNodeProto(this, 'Pad', padOnnxName, padInput, padOutput, pads, padOnnxMode, paddingValue);

                nodeProto(end+1) = padNode;

                % update conv input: padOnnxName is the input and padding
                % has size zero.
                convOperatorInput = padOnnxName;
                permutedPadding   = 0*permutedPadding;
            end

            % Make the nodeProto
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            outputTensorLayout      = inputTensorLayout;

            WName                   = [onnxName '_W'];
            BName                   = [onnxName '_B'];
            newNode                 = NodeProto;
            newNode.op_type         = 'Conv';
            newNode.name            = onnxName;
            newNode.input           = mapTensorNames(this, ...
                {convOperatorInput, WName, BName}, TensorNameMap);
            newNode.output          = {onnxName};

            newNode.attribute       = [...
                makeAttributeProto('group',        'INT',  this.NNTLayer.NumGroups),...
                makeAttributeProto('dilations',    'INTS', this.NNTLayer.DilationFactor),...
                makeAttributeProto('kernel_shape', 'INTS', this.NNTLayer.FilterSize),...
                makeAttributeProto('pads',         'INTS', permutedPadding),...
                makeAttributeProto('strides',      'INTS', this.NNTLayer.Stride)
                ];

            % Make parameter Initializers for: W, B
            t1              = TensorProto;
            t1.name         = WName;
            t1.data_type	= TensorProto_DataType.FLOAT;
            internalWeights = reshape(this.NNTLayer.Weights, [this.NNTLayer.FilterSize...
                this.NNTLayer.NumChannelsPerGroup, ...
                this.NNTLayer.NumGroups * this.NNTLayer.NumFiltersPerGroup]);
            permutedW       = permute(internalWeights, [4 3 1 2]);	% NNT is HWCF. ONNX is FCHW.
            t1.raw_data     = rawData(single(permutedW));
            t1.dims         = dimVector(size(permutedW),4);

            t2              = TensorProto;
            t2.name         = BName;
            t2.data_type    = TensorProto_DataType.FLOAT;
            internalBias = reshape(this.NNTLayer.Bias, [1, 1, ...
                this.NNTLayer.NumGroups * this.NNTLayer.NumFiltersPerGroup]);
            t2.raw_data     = rawData(single(squeeze(internalBias)));
            t2.dims         = dimVector(numel(this.NNTLayer.Bias),1);            % NNT data: 1-1-numFilters

            parameterInitializers = [padInits t1 t2];


            nodeProto(end+1)        = newNode;
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            TensorNameMap(this.NNTLayer.Name) = onnxName;
            TensorLayoutMap(onnxName) = outputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

