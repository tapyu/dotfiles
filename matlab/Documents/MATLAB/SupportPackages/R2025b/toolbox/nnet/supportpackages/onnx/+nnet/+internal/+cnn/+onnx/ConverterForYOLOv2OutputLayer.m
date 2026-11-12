classdef ConverterForYOLOv2OutputLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a yolov2OutputLayer into ONNX.

    % Copyright 2019-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForYOLOv2OutputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            existingNodeNames            = {nodeProto.name};
            [onnxName, ~]                = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames             = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);

            % (1) Reshape input X into [BatchSize, predictionsPerAnchor,
            % numAnchors, numGrids]
            InputLayerSize               = this.InputLayerSizes{1};
            XH                           = InputLayerSize(1);
            XW                           = InputLayerSize(2);
            XC                           = InputLayerSize(3);
            XA                           = size(this.NNTLayer.AnchorBoxes,1);
            shape                        = [-1, XC/XA, XA, XH*XW];  
            shapeTensor(1)               = TensorProto;
            shapeTensor(1).name          = [onnxName '_Reshape1_shape'];
            shapeTensor(1).data_type     = TensorProto_DataType.INT64;
            shapeTensor(1).raw_data      = rawData(int64(shape));
            shapeTensor(1).dims          = dimVector(numel(shape),1);
            
            reshapeNodeName              = [onnxName '_Reshape1'];
            reshapeNodeName              = makeUniqueName(existingNodeNames, reshapeNodeName);
            reshapeNode(1)               = NodeProto;
            reshapeNode(1).op_type       = 'Reshape';
            reshapeNode(1).name          = reshapeNodeName;
            reshapeNode(1).input         = {inputTensorNames{1,1}, shapeTensor(1).name};
            reshapeNode(1).output        = {reshapeNodeName};
            parameterInitializers        = shapeTensor;

            outputTensorName  = reshapeNodeName;
            outputTensorSize  = [{this.BatchSizeToExport}, {XC/XA, XA, XH*XW}];	% [BatchSize, predictionsPerAnchor, numAnchors, numGrids]
            
            nodeProto               = [nodeProto reshapeNode];
            networkInputs           = [];
            networkOutputs          = makeValueInfoProtoFromDimensions(outputTensorName, TensorProto_DataType.FLOAT, outputTensorSize);
        end
    end
end
