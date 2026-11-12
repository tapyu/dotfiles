classdef ConverterForUnsupportedLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert an unknown layer into an ONNX
    % com.mathworks.Placeholder operator.
    %
    % For example:
    % nodeProto =
    %   NodeProto with properties:
    %          input: {'fc1'  'fc2'  'fc3'  'fc4'}
    %         output: {'add4to2_out1'  'add4to2_out2'}
    %           name: 'add4to2'
    %        op_type: 'Placeholder'
    %         domain: 'com.mathworks'
    %      attribute: []
    %     doc_string: 'Placeholder operator'
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    properties
        LayerAnalyzer
    end
    
    methods
        function this = ConverterForUnsupportedLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
            this.LayerAnalyzer = layerAnalyzer;
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % An NNT unknown layer translates into a com.mathworks.Placeholder operator.
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            
            % Output names are the output tensors of the layer. Prepend the
            % layer name and '/'. If there is only one output, use the
            % layer name alone. Finally, legalize the names for ONNX.
            OutputNames             = this.LayerAnalyzer.Outputs.Properties.RowNames(:)';
            if numel(OutputNames)==1
                OutputNames         = {this.NNTLayer.Name};
            else
                OutputNames       	= cellfun(@(outName)[this.NNTLayer.Name '/' outName], OutputNames, 'UniformOutput', false);
            end
            OutputONNXNames         = cellfun(@(name)legalizeNNTName(this, name), OutputNames, 'UniformOutput', false);
            
            % Make the nodeProto
            newNode               = NodeProto;
            newNode.op_type       = 'Placeholder';
            newNode.domain        = 'com.mathworks';
            newNode.name          = onnxName;
            newNode.input         = inputTensorNames;
            newNode.output        = OutputONNXNames;
            newNode.doc_string	= getString(message('nnet_cnn_onnx:onnx:PlaceholderOperatorDocString'));
            
            nodeProto(end+1)        = newNode;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];
            
            % Add output names to maps
            for i=1:numel(OutputNames)
                TensorNameMap(OutputNames{i})         = OutputONNXNames{i};
                % Extract output labels for the unsupported layer and
                % update the tensor layout map
                TensorLayoutMap(OutputONNXNames{i})   = convertToONNXLabel(this, this.LayerAnalyzer.Outputs.Meta{i}.dims);
            end
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end

        % Converts DLT data formats to sorted ONNX format
        function ONNXLabels = convertToONNXLabel(this, DLTLabels)
            DLTLabels = char(DLTLabels);
            
            DLT2ONNX = createDLT2ONNXDict;
            if isKey(DLT2ONNX, DLTLabels)
                ONNXLabels = DLT2ONNX(DLTLabels);
            else
                error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
        end        
    end
end

function DLT2ONNX = createDLT2ONNXDict
    DLT2ONNX = dictionary;

    % Feature Tensors
    DLT2ONNX('CB') = 'nc';
    DLT2ONNX('CU') = 'nc';
    % Spational Tensors
    DLT2ONNX('SC') = 'nch';
    DLT2ONNX('SSC') = 'nchw';
    DLT2ONNX('SSSC') = 'nchwd';
    DLT2ONNX('SCB') = 'nch';
    DLT2ONNX('SSCB') = 'nchw';
    DLT2ONNX('SSSCB') = 'nchwd';
    % Temporal Tensors
    DLT2ONNX('CT') = 'snc';
    DLT2ONNX('CBT') = 'snc';  
    % Spatial+Temporal Tensors
    DLT2ONNX('SCT') = 'snch';
    DLT2ONNX('SSCT') = 'snchw';
    DLT2ONNX('SSSCT') = 'snchwd';
    DLT2ONNX('SCBT') = 'snch';
    DLT2ONNX('SSCBT') = 'snchw';
    DLT2ONNX('SSSCBT') = 'snchwd';  
end