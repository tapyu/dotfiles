classdef ConverterForNASNetMobileCrop2dLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.nasnetmobile.layer.NASNetMobileCrop2dLayer into ONNX

    % Copyright 2019-2024 The MathWorks, Inc.

    methods
        function this = ConverterForNASNetMobileCrop2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            outputTensorLayout      = inputTensorLayout;

            % Find Left, Top, Right, Bottom of resulting region
            XH                  = this.InputLayerSizes{1}(1);
            XW                  = this.InputLayerSizes{1}(2);
            ReferenceLayerSize  = this.InputLayerSizes{2};
            H                   = ReferenceLayerSize(1);
            W                   = ReferenceLayerSize(2);
            assert(isnumeric(this.NNTLayer.Location) && numel(this.NNTLayer.Location)==2,...
                message('nnet_cnn_onnx:onnx:UnsupportedCropLocation', this.NNTLayer.Name)); % [L T]
            T = this.NNTLayer.Location(2);
            L = this.NNTLayer.Location(1);
            B = T + H - 1;
            R = L + W - 1;

            % Convert from retained region to borders (number of rows and cols to delete):
            LB = L - 1;
            TB = T - 1;
            RB = XW - R;
            BB = XH - B;

            % RB or BB may be zero if there are no rows/cols to delete.
            % In this case, end should be intmax (retain the entire
            % input along the axis)
            endR = -RB;
            if endR == 0
                endR = intmax;
            end

            endB = -BB;
            if endB == 0
                endB = intmax;
            end

            % Create Slice node
            sliceName           = onnxName;
            sliceInput          = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            sliceOutput         = {onnxName};
            starts              = [0 0 TB LB];
            ends                = [intmax intmax endB endR];
            [sliceNode, inits]  = createNodeProto(this, 'Slice', sliceName, sliceInput, sliceOutput, starts, ends);

            nodeProto(end+1)        = sliceNode;
            parameterInitializers   = inits;
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            outputTensorName = addNodeName;
            TensorNameMap(this.NNTLayer.Name) = onnxName;
            TensorLayoutMap(outputTensorName) = outputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
