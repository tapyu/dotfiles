classdef TypeProto_Opaque
%     optional string domain = 1;
%     optional string name = 2;
    
    %   Copyright 2019-2021 The MathWorks, Inc.

    properties
        domain
        name
    end
    
    methods
        function this = TypeProto_Opaque(varargin)
            import nnet.internal.cnn.onnx.*
            %Add Spkg bin path to system path
            nnet.internal.cnn.onnx.util.addSpkgBinPath();
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTypeProto_Opaque), Ptr);
                [this.domain, this.name] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                % (none)
            end
        end
        
        function encodeTypeProto_Opaque(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.domain, this.name};
            PtrCell = onnxmex(int32(FuncName.EencodeTypeProto_Opaque), CPtr, PropertyCell);
            % Fill pointer objects
            % (none)
        end
    end
end