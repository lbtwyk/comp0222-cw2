classdef RelativePoseSE2Edge < g2o.core.BaseBinaryEdge
    %RELATIVEPOSESE2EDGE Binary SE(2) relative-pose constraint.

    methods(Access = public)
        function obj = RelativePoseSE2Edge()
            obj = obj@g2o.core.BaseBinaryEdge(3);
        end

        function initialEstimate(obj)
            v1 = obj.edgeVertices{1};
            v2 = obj.edgeVertices{2};

            x1 = v1.estimate();
            z = obj.measurement();
            v2.setEstimate(q3d.RelativePoseSE2Edge.composePose(x1, z));
        end

        function computeError(obj)
            v1 = obj.edgeVertices{1};
            v2 = obj.edgeVertices{2};

            prediction = q3d.RelativePoseSE2Edge.relativePose(v1.estimate(), v2.estimate());
            obj.errorZ = prediction - obj.measurement();
            obj.errorZ(3) = g2o.stuff.normalize_theta(obj.errorZ(3));
        end

        function linearizeOplus(obj)
            v1 = obj.edgeVertices{1};
            v2 = obj.edgeVertices{2};

            x1 = v1.estimate();
            x2 = v2.estimate();
            basePrediction = q3d.RelativePoseSE2Edge.relativePose(x1, x2);

            obj.J{1} = q3d.RelativePoseSE2Edge.numericJacobian(x1, x2, basePrediction, 1);
            obj.J{2} = q3d.RelativePoseSE2Edge.numericJacobian(x1, x2, basePrediction, 2);
        end
    end

    methods(Access = private, Static)
        function J = numericJacobian(x1, x2, basePrediction, vertexNumber)
            epsilon = 1e-6;
            J = zeros(3, 3);

            for idx = 1 : 3
                if vertexNumber == 1
                    xp1 = x1;
                    xp2 = x2;
                    xp1(idx) = xp1(idx) + epsilon;
                    xp1(3) = g2o.stuff.normalize_theta(xp1(3));
                else
                    xp1 = x1;
                    xp2 = x2;
                    xp2(idx) = xp2(idx) + epsilon;
                    xp2(3) = g2o.stuff.normalize_theta(xp2(3));
                end

                prediction = q3d.RelativePoseSE2Edge.relativePose(xp1, xp2);
                delta = prediction - basePrediction;
                delta(3) = g2o.stuff.normalize_theta(delta(3));
                J(:, idx) = delta / epsilon;
            end
        end
    end

    methods(Access = public, Static)
        function x2 = composePose(x1, z)
            c = cos(x1(3));
            s = sin(x1(3));

            x2 = zeros(3, 1);
            x2(1) = x1(1) + c * z(1) - s * z(2);
            x2(2) = x1(2) + s * z(1) + c * z(2);
            x2(3) = g2o.stuff.normalize_theta(x1(3) + z(3));
        end

        function z = relativePose(x1, x2)
            dx = x2(1) - x1(1);
            dy = x2(2) - x1(2);
            c = cos(x1(3));
            s = sin(x1(3));

            z = zeros(3, 1);
            z(1) = c * dx + s * dy;
            z(2) = -s * dx + c * dy;
            z(3) = g2o.stuff.normalize_theta(x2(3) - x1(3));
        end
    end
end
