classdef FunctionProto
    % Language binding class for the FunctionProto protobuf message
    % optional string name = 1;
    % repeated string input = 4;
    % repeated string output = 5;
    % repeated string attribute = 6;
    % repeated AttributeProto attribute_proto = 11;
    % repeated NodeProto node = 7;
    % optional string doc_string = 8;
    % repeated OperatorSetIdProto opset_import = 9;
    % optional string domain = 10;

    %   Copyright 2021-2024 The MathWorks, Inc.
    properties
        name
        input
        output
        attribute
        attribute_proto
        node
        doc_string
        opset_import
        domain
    end
    
    methods
        function this = FunctionProto(varargin)
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeFunctionProto), Ptr);
                [this.name, this.input, this.output, this.attribute, this.attribute_proto, ...
                    this.node, this.doc_string, this.opset_import, this.domain] = PropertyCell{:};
                % Call constructors on Proto objects
                if ~isempty(this.attribute_proto)
                    this.attribute_proto = arrayfun(@AttributeProto, this.attribute_proto);
                end
                if ~isempty(this.node)
                    this.node = arrayfun(@NodeProto, this.node);
                end
                if ~isempty(this.opset_import)
                    this.opset_import = arrayfun(@OperatorSetIdProto, this.opset_import);
                end
            end
        end
        
        function encodeFunctionProto(this, CPtr)
             % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.name, this.input, this.output, this.attribute, ...
                    this.attribute_proto, this.node, this.doc_string, this.opset_import, this.domain};
            PtrCell = onnxmex(int32(FuncName.EencodeFunctionProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeAttributeProto, this.attribute_proto, PtrCell{5});
            arrayfun(@encodeNodeProto, this.node, PtrCell{6});     
            arrayfun(@encodeOperatorSetIdProto, this.opset_import, PtrCell{8});                 
        end
    end
end
