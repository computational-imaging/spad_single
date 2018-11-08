% mex -I/usr/local/include/ANN -L/usr/lib/ -lann -I/usr/local/include/opencv -lopencv_core getConstraintsMatrix.cpp mexBase.cpp
% mex -I/usr/local/include/ANN -L/usr/lib/ -lann -I/usr/local/include/opencv -lopencv_core getContinuousConstraintMatrix.cpp mexBase.cpp
% mex -I/usr/local/include/ANN -L/usr/lib/ -lann -I/usr/local/include/opencv -lopencv_core getGridLLEMatrix.cpp mexBase.cpp LLE.cpp
% mex -I/usr/local/include/ANN -L/usr/lib/ -lann -I/usr/local/include/opencv -lopencv_core getGridLLEMatrixNormal.cpp mexBase.cpp LLE.cpp
% mex -I/usr/local/include/ANN -L/usr/lib/ -lann -I/usr/local/include/opencv -lopencv_core getNormalConstraintMatrix.cpp mexBase.cpp
mex -I/usr/local/include/ -lopencv_core getConstraintsMatrix.cpp mexBase.cpp
mex -I/usr/local/include/ -lopencv_core getContinuousConstraintMatrix.cpp mexBase.cpp
mex -Iann_1.1.2/include -Lann_1.1.2/lib -lann -I/usr/local/include/ -lopencv_core getGridLLEMatrix.cpp mexBase.cpp LLE.cpp
mex -Iann_1.1.2/include -Lann_1.1.2/lib -lann -I/usr/local/include/ -lopencv_core getGridLLEMatrixNormal.cpp mexBase.cpp LLE.cpp
mex -Iann_1.1.2/include -Lann_1.1.2/lib -lann -I/usr/local/include/ -lopencv_core getNormalConstraintMatrix.cpp mexBase.cpp

