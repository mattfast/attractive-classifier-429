��	
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02unknown8��
~
conv2d/kernelVarHandleOp*
shared_nameconv2d/kernel*
_output_shapes
: *
dtype0*
shape:`
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:`*
dtype0
n
conv2d/biasVarHandleOp*
dtype0*
shape:`*
_output_shapes
: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:`
�
conv2d_1/kernelVarHandleOp*
shape:`�* 
shared_nameconv2d_1/kernel*
dtype0*
_output_shapes
: 
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*'
_output_shapes
:`�
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
shape:�*
shared_nameconv2d_1/bias*
dtype0
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:�*
dtype0
�
conv2d_2/kernelVarHandleOp* 
shared_nameconv2d_2/kernel*
dtype0*
_output_shapes
: *
shape:��
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*(
_output_shapes
:��
s
conv2d_2/biasVarHandleOp*
shape:�*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:�*
dtype0
u
dense/kernelVarHandleOp*
shape:	�@*
shared_namedense/kernel*
_output_shapes
: *
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	�@
l

dense/biasVarHandleOp*
shared_name
dense/bias*
dtype0*
shape:@*
_output_shapes
: 
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:@
x
dense_1/kernelVarHandleOp*
shared_namedense_1/kernel*
shape
:@*
_output_shapes
: *
dtype0
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
shared_namedense_1/bias*
_output_shapes
: *
dtype0*
shape:
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
shape: *
_output_shapes
: *
dtype0	*
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
shape: *
dtype0*
shared_nameAdam/beta_1*
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
dtype0*
shared_nameAdam/beta_2*
shape: *
_output_shapes
: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
_output_shapes
: *
shared_name
Adam/decay*
dtype0*
shape: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shape: *
_output_shapes
: *
shared_nametotal*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
_output_shapes
: *
dtype0*
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
�
Adam/conv2d/kernel/mVarHandleOp*%
shared_nameAdam/conv2d/kernel/m*
_output_shapes
: *
shape:`*
dtype0
�
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*
dtype0*&
_output_shapes
:`
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *#
shared_nameAdam/conv2d/bias/m*
dtype0*
shape:`
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
dtype0*
_output_shapes
:`
�
Adam/conv2d_1/kernel/mVarHandleOp*
shape:`�*'
shared_nameAdam/conv2d_1/kernel/m*
dtype0*
_output_shapes
: 
�
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*
dtype0*'
_output_shapes
:`�
�
Adam/conv2d_1/bias/mVarHandleOp*%
shared_nameAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0*
shape:�
z
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_2/kernel/mVarHandleOp*'
shared_nameAdam/conv2d_2/kernel/m*
_output_shapes
: *
dtype0*
shape:��
�
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*
dtype0*(
_output_shapes
:��
�
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *%
shared_nameAdam/conv2d_2/bias/m*
dtype0*
shape:�
z
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
dtype0*
_output_shapes	
:�
�
Adam/dense/kernel/mVarHandleOp*
shape:	�@*
dtype0*$
shared_nameAdam/dense/kernel/m*
_output_shapes
: 
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
dtype0*
_output_shapes
:	�@
z
Adam/dense/bias/mVarHandleOp*
shape:@*"
shared_nameAdam/dense/bias/m*
dtype0*
_output_shapes
: 
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
dtype0*
_output_shapes
:@
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*&
shared_nameAdam/dense_1/kernel/m*
shape
:@

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
shape:*$
shared_nameAdam/dense_1/bias/m*
dtype0
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d/kernel/vVarHandleOp*
shape:`*
_output_shapes
: *%
shared_nameAdam/conv2d/kernel/v*
dtype0
�
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:`*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
shape:`*
_output_shapes
: *#
shared_nameAdam/conv2d/bias/v*
dtype0
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:`*
dtype0
�
Adam/conv2d_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *'
shared_nameAdam/conv2d_1/kernel/v*
shape:`�
�
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*
dtype0*'
_output_shapes
:`�
�
Adam/conv2d_1/bias/vVarHandleOp*
shape:�*%
shared_nameAdam/conv2d_1/bias/v*
dtype0*
_output_shapes
: 
z
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
shape:��*
dtype0*'
shared_nameAdam/conv2d_2/kernel/v
�
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*
dtype0*(
_output_shapes
:��
�
Adam/conv2d_2/bias/vVarHandleOp*
dtype0*
_output_shapes
: *%
shared_nameAdam/conv2d_2/bias/v*
shape:�
z
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*$
shared_nameAdam/dense/kernel/v*
shape:	�@
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
dtype0*
_output_shapes
:	�@
z
Adam/dense/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
dtype0*
_output_shapes
:@
�
Adam/dense_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *&
shared_nameAdam/dense_1/kernel/v*
shape
:@

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
dtype0*
_output_shapes

:@
~
Adam/dense_1/bias/vVarHandleOp*
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v*
_output_shapes
: 
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�@
ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *�?
value�?B�? B�?
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
R
%trainable_variables
&regularization_losses
'	variables
(	keras_api
h

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
2	keras_api
R
3trainable_variables
4regularization_losses
5	variables
6	keras_api
h

7kernel
8bias
9trainable_variables
:regularization_losses
;	variables
<	keras_api
h

=kernel
>bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
�
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem�m�m� m�)m�*m�7m�8m�=m�>m�v�v�v� v�)v�*v�7v�8v�=v�>v�
F
0
1
2
 3
)4
*5
76
87
=8
>9
 
F
0
1
2
 3
)4
*5
76
87
=8
>9
�

Hlayers
Ilayer_regularization_losses
trainable_variables
Jnon_trainable_variables
Kmetrics
regularization_losses
	variables
 
 
 
 
�

Llayers
Mlayer_regularization_losses
trainable_variables
Nnon_trainable_variables
Ometrics
regularization_losses
	variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

Players
Qlayer_regularization_losses
trainable_variables
Rnon_trainable_variables
Smetrics
regularization_losses
	variables
 
 
 
�

Tlayers
Ulayer_regularization_losses
trainable_variables
Vnon_trainable_variables
Wmetrics
regularization_losses
	variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
�

Xlayers
Ylayer_regularization_losses
!trainable_variables
Znon_trainable_variables
[metrics
"regularization_losses
#	variables
 
 
 
�

\layers
]layer_regularization_losses
%trainable_variables
^non_trainable_variables
_metrics
&regularization_losses
'	variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
�

`layers
alayer_regularization_losses
+trainable_variables
bnon_trainable_variables
cmetrics
,regularization_losses
-	variables
 
 
 
�

dlayers
elayer_regularization_losses
/trainable_variables
fnon_trainable_variables
gmetrics
0regularization_losses
1	variables
 
 
 
�

hlayers
ilayer_regularization_losses
3trainable_variables
jnon_trainable_variables
kmetrics
4regularization_losses
5	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81
 

70
81
�

llayers
mlayer_regularization_losses
9trainable_variables
nnon_trainable_variables
ometrics
:regularization_losses
;	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
 

=0
>1
�

players
qlayer_regularization_losses
?trainable_variables
rnon_trainable_variables
smetrics
@regularization_losses
A	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
	7

8
 
 

t0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	utotal
	vcount
w
_fn_kwargs
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

u0
v1
�

|layers
}layer_regularization_losses
xtrainable_variables
~non_trainable_variables
metrics
yregularization_losses
z	variables
 
 

u0
v1
 
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_conv2d_inputPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tout
2**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_4372*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-4617*
Tin
2
O
saver_filenamePlaceholder*
dtype0*
shape: *
_output_shapes
: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*+
_gradient_op_typePartitionedCall-4676**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *
Tout
2*&
f!R
__inference__traced_save_4675*2
Tin+
)2'	
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*
Tout
2*+
_gradient_op_typePartitionedCall-4800*)
f$R"
 __inference__traced_restore_4799*1
Tin*
(2&*
_output_shapes
: **
config_proto

CPU

GPU 2J 8��
�5
�
D__inference_sequential_layer_call_and_return_conditional_losses_4462

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:`*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:���������{{`*
T0*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:`*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������{{`f
conv2d/ReluReluconv2d/BiasAdd:output:0*/
_output_shapes
:���������{{`*
T0�
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*
paddingVALID*
strides
*
ksize
*/
_output_shapes
:���������==`�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:`��
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:����������*
strides
*
T0*
paddingVALID�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:����������*
T0k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*0
_output_shapes
:����������*
T0�
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
paddingVALID*
ksize
*
strides
*0
_output_shapes
:�����������
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*0
_output_shapes
:�����������
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:����������*
T0k
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
f
flatten/Reshape/shapeConst*
valueB"����   *
_output_shapes
:*
dtype0�
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Reshape/shape:output:0*(
_output_shapes
:����������*
T0�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	�@*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitydense_1/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp: : :	 :
 :& "
 
_user_specified_nameinputs: : : : : : 
�	
�
A__inference_dense_1_layer_call_and_return_conditional_losses_4532

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4100

inputs
identity�
MaxPoolMaxPoolinputs*
paddingVALID*
ksize
*
strides
*J
_output_shapes8
6:4������������������������������������{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
@__inference_conv2d_layer_call_and_return_conditional_losses_4039

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:`�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*A
_output_shapes/
-:+���������������������������`*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:`�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+���������������������������`*
T0j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������`�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+���������������������������`*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
'__inference_conv2d_2_layer_call_fn_4134

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-4129*
Tin
2*B
_output_shapes0
.:,����������������������������*
Tout
2*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4123�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,����������������������������*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�	
�
?__inference_dense_layer_call_and_return_conditional_losses_4514

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0P
ReluReluBiasAdd:output:0*'
_output_shapes
:���������@*
T0�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������@*
T0"
identityIdentity:output:0*/
_input_shapes
:����������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
$__inference_dense_layer_call_fn_4521

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:���������@*
Tin
2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4199**
config_proto

CPU

GPU 2J 8*
Tout
2*+
_gradient_op_typePartitionedCall-4205�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������@*
T0"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
)__inference_sequential_layer_call_fn_4351
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*+
_gradient_op_typePartitionedCall-4338*
Tin
2*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4337*
Tout
2*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : :	 :
 
�
J
.__inference_max_pooling2d_2_layer_call_fn_4151

inputs
identity�
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-4148**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4142*
Tout
2*J
_output_shapes8
6:4������������������������������������*
Tin
2�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
'__inference_conv2d_1_layer_call_fn_4092

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4081*B
_output_shapes0
.:,����������������������������*
Tout
2*+
_gradient_op_typePartitionedCall-4087�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,����������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������`::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
)__inference_sequential_layer_call_fn_4310
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tout
2*
Tin
2*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4296*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-4297**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : :	 :
 :, (
&
_user_specified_nameconv2d_input
�	
�
?__inference_dense_layer_call_and_return_conditional_losses_4199

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*'
_output_shapes
:���������@*
T0�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�'
�
D__inference_sequential_layer_call_and_return_conditional_losses_4245
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_4039*
Tin
2*+
_gradient_op_typePartitionedCall-4045*/
_output_shapes
:���������{{`**
config_proto

CPU

GPU 2J 8�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4058*
Tout
2*+
_gradient_op_typePartitionedCall-4064*/
_output_shapes
:���������==`**
config_proto

CPU

GPU 2J 8�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2*0
_output_shapes
:����������*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4081*+
_gradient_op_typePartitionedCall-4087�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tout
2*+
_gradient_op_typePartitionedCall-4106*
Tin
2**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:����������*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4100�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4129*0
_output_shapes
:����������*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4123*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2�
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-4148*R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4142*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8�
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4175*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*
Tout
2*+
_gradient_op_typePartitionedCall-4181�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������@*
Tout
2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4199*+
_gradient_op_typePartitionedCall-4205*
Tin
2�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tout
2*+
_gradient_op_typePartitionedCall-4233*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4227*'
_output_shapes
:���������*
Tin
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : :	 :
 
�
B
&__inference_flatten_layer_call_fn_4503

inputs
identity�
PartitionedCallPartitionedCallinputs*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4175*+
_gradient_op_typePartitionedCall-4181*
Tout
2*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*
Tin
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4081

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*'
_output_shapes
:`�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
paddingVALID*B
_output_shapes0
.:,����������������������������*
strides
�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,����������������������������*
T0k
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
T0�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������`::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
J
.__inference_max_pooling2d_1_layer_call_fn_4109

inputs
identity�
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-4106*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4100*J
_output_shapes8
6:4������������������������������������*
Tout
2*
Tin
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
)__inference_sequential_layer_call_fn_4492

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*+
_gradient_op_typePartitionedCall-4338*
Tout
2**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4337*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 
�	
�
A__inference_dense_1_layer_call_and_return_conditional_losses_4227

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4123

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*B
_output_shapes0
.:,����������������������������*
strides
*
T0*
paddingVALID�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,����������������������������*
T0k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,�����������������������������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4058

inputs
identity�
MaxPoolMaxPoolinputs*
ksize
*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4������������������������������������*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�&
�
D__inference_sequential_layer_call_and_return_conditional_losses_4296

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_4039*
Tout
2**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-4045*
Tin
2*/
_output_shapes
:���������{{`�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4058*
Tout
2*
Tin
2*/
_output_shapes
:���������==`*+
_gradient_op_typePartitionedCall-4064�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4081*+
_gradient_op_typePartitionedCall-4087*0
_output_shapes
:����������*
Tout
2�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tout
2*0
_output_shapes
:����������*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4100*
Tin
2*+
_gradient_op_typePartitionedCall-4106�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*0
_output_shapes
:����������*
Tout
2*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4123*+
_gradient_op_typePartitionedCall-4129*
Tin
2**
config_proto

CPU

GPU 2J 8�
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tout
2*R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4142*
Tin
2*+
_gradient_op_typePartitionedCall-4148**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:�����������
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-4181*
Tin
2*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4175**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:����������*
Tout
2�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2*'
_output_shapes
:���������@**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-4205*
Tin
2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4199�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4233**
config_proto

CPU

GPU 2J 8*
Tout
2*'
_output_shapes
:���������*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4227*
Tin
2�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall: : : : :	 :
 :& "
 
_user_specified_nameinputs: : : : 
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_4175

inputs
identity^
Reshape/shapeConst*
valueB"����   *
_output_shapes
:*
dtype0e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:����������*
T0Y
IdentityIdentityReshape:output:0*(
_output_shapes
:����������*
T0"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4142

inputs
identity�
MaxPoolMaxPoolinputs*
ksize
*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4������������������������������������*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
H
,__inference_max_pooling2d_layer_call_fn_4067

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4058*J
_output_shapes8
6:4������������������������������������*+
_gradient_op_typePartitionedCall-4064*
Tin
2�
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4������������������������������������*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�?
�
__inference__wrapped_model_4025
conv2d_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identity��(sequential/conv2d/BiasAdd/ReadVariableOp�'sequential/conv2d/Conv2D/ReadVariableOp�*sequential/conv2d_1/BiasAdd/ReadVariableOp�)sequential/conv2d_1/Conv2D/ReadVariableOp�*sequential/conv2d_2/BiasAdd/ReadVariableOp�)sequential/conv2d_2/Conv2D/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:`*
dtype0�
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������{{`*
strides
*
paddingVALID�
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:`*
dtype0�
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:���������{{`*
T0|
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������{{`�
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*
ksize
*/
_output_shapes
:���������==`*
strides
*
paddingVALID�
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:`��
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
paddingVALID*0
_output_shapes
:����������*
strides
�
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
"sequential/max_pooling2d_1/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*
paddingVALID*
ksize
*
strides
*0
_output_shapes
:�����������
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:���
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
paddingVALID*
T0*0
_output_shapes
:����������*
strides
�
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:����������*
T0�
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*0
_output_shapes
:����������*
T0�
"sequential/max_pooling2d_2/MaxPoolMaxPool&sequential/conv2d_2/Relu:activations:0*0
_output_shapes
:����������*
paddingVALID*
strides
*
ksize
q
 sequential/flatten/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:�
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_2/MaxPool:output:0)sequential/flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�@�
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@*
dtype0�
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0|
sequential/dense_1/SigmoidSigmoid#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitysequential/dense_1/Sigmoid:y:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp: : : : : : : : :	 :
 :, (
&
_user_specified_nameconv2d_input
�
�
)__inference_sequential_layer_call_fn_4477

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tout
2*+
_gradient_op_typePartitionedCall-4297*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4296*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : :	 :
 :& "
 
_user_specified_nameinputs
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_4498

inputs
identity^
Reshape/shapeConst*
valueB"����   *
_output_shapes
:*
dtype0e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�H
�
__inference__traced_save_4675
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_1e99e54e19ab4ab8a2039ef12b32c70b/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
value	B :*
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*�
value�B�%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0�
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:%*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop"/device:CPU:0*3
dtypes)
'2%	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
value	B :*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
N�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :`:`:`�:�:��:�:	�@:@:@:: : : : : : : :`:`:`�:�:��:�:	�@:@:@::`:`:`�:�:��:�:	�@:@:@:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : 
�
�
"__inference_signature_wrapper_4372
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tout
2*(
f#R!
__inference__wrapped_model_4025*+
_gradient_op_typePartitionedCall-4359**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : :	 :
 
�5
�
D__inference_sequential_layer_call_and_return_conditional_losses_4418

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:`*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
paddingVALID*
strides
*
T0*/
_output_shapes
:���������{{`�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:`�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������{{`f
conv2d/ReluReluconv2d/BiasAdd:output:0*/
_output_shapes
:���������{{`*
T0�
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*
strides
*
ksize
*/
_output_shapes
:���������==`*
paddingVALID�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:`��
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:����������*
T0*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:����������*
T0k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
strides
*0
_output_shapes
:����������*
ksize
*
paddingVALID�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*0
_output_shapes
:����������*
paddingVALID*
strides
*
T0�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:����������*
T0k
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*0
_output_shapes
:����������*
T0�
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
strides
*
paddingVALIDf
flatten/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:�
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�@�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*'
_output_shapes
:���������@*
T0�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:@*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitydense_1/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp:	 :
 :& "
 
_user_specified_nameinputs: : : : : : : : 
�
�
&__inference_dense_1_layer_call_fn_4539

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:���������*
Tin
2*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4227**
config_proto

CPU

GPU 2J 8*
Tout
2*+
_gradient_op_typePartitionedCall-4233�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
%__inference_conv2d_layer_call_fn_4050

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_4039*A
_output_shapes/
-:+���������������������������`*
Tin
2*+
_gradient_op_typePartitionedCall-4045*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������`"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�'
�
D__inference_sequential_layer_call_and_return_conditional_losses_4270
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*/
_output_shapes
:���������{{`**
config_proto

CPU

GPU 2J 8*
Tin
2*
Tout
2*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_4039*+
_gradient_op_typePartitionedCall-4045�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-4064*
Tout
2*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4058**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:���������==`*
Tin
2�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4087*0
_output_shapes
:����������*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4081�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*0
_output_shapes
:����������*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4100*
Tout
2**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-4106*
Tin
2�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4129*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4123**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:����������*
Tin
2*
Tout
2�
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tout
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-4148**
config_proto

CPU

GPU 2J 8*
Tin
2*R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4142�
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4175**
config_proto

CPU

GPU 2J 8*
Tout
2*
Tin
2*(
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-4181�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4205*'
_output_shapes
:���������@*
Tin
2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4199*
Tout
2**
config_proto

CPU

GPU 2J 8�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*'
_output_shapes
:���������*+
_gradient_op_typePartitionedCall-4233**
config_proto

CPU

GPU 2J 8*
Tout
2*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4227*
Tin
2�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall: : : : : : : : :	 :
 :, (
&
_user_specified_nameconv2d_input
ō
�
 __inference__traced_restore_4799
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias%
!assignvariableop_8_dense_1_kernel#
assignvariableop_9_dense_1_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count,
(assignvariableop_17_adam_conv2d_kernel_m*
&assignvariableop_18_adam_conv2d_bias_m.
*assignvariableop_19_adam_conv2d_1_kernel_m,
(assignvariableop_20_adam_conv2d_1_bias_m.
*assignvariableop_21_adam_conv2d_2_kernel_m,
(assignvariableop_22_adam_conv2d_2_bias_m+
'assignvariableop_23_adam_dense_kernel_m)
%assignvariableop_24_adam_dense_bias_m-
)assignvariableop_25_adam_dense_1_kernel_m+
'assignvariableop_26_adam_dense_1_bias_m,
(assignvariableop_27_adam_conv2d_kernel_v*
&assignvariableop_28_adam_conv2d_bias_v.
*assignvariableop_29_adam_conv2d_1_kernel_v,
(assignvariableop_30_adam_conv2d_1_bias_v.
*assignvariableop_31_adam_conv2d_2_kernel_v,
(assignvariableop_32_adam_conv2d_2_bias_v+
'assignvariableop_33_adam_dense_kernel_v)
%assignvariableop_34_adam_dense_bias_v-
)assignvariableop_35_adam_dense_1_kernel_v+
'assignvariableop_36_adam_dense_1_bias_v
identity_38��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:%�
RestoreV2/shape_and_slicesConst"/device:CPU:0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:%�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:z
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:~
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:}
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0*
dtype0	*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0{
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:{
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_conv2d_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_conv2d_bias_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_1_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_1_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype0P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_2_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_2_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype0P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_1_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype0P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_1_bias_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv2d_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype0P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv2d_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype0P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_1_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype0P
Identity_30IdentityRestoreV2:tensors:30*
_output_shapes
:*
T0�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_1_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype0P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_2_kernel_vIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
_output_shapes
:*
T0�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_2_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype0P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_kernel_vIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
_output_shapes
:*
T0�
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_dense_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype0P
Identity_35IdentityRestoreV2:tensors:35*
_output_shapes
:*
T0�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_1_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype0P
Identity_36IdentityRestoreV2:tensors:36*
_output_shapes
:*
T0�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_1_bias_vIdentity_36:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0�
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_38Identity_38:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5: : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : 
�&
�
D__inference_sequential_layer_call_and_return_conditional_losses_4337

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_4039**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������{{`*+
_gradient_op_typePartitionedCall-4045�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:���������==`*
Tout
2*+
_gradient_op_typePartitionedCall-4064*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4058*
Tin
2�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*0
_output_shapes
:����������*
Tout
2*+
_gradient_op_typePartitionedCall-4087*
Tin
2**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4081�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tout
2*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4100**
config_proto

CPU

GPU 2J 8*+
_gradient_op_typePartitionedCall-4106*0
_output_shapes
:����������*
Tin
2�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4123*+
_gradient_op_typePartitionedCall-4129*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2*0
_output_shapes
:�����������
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*0
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*
Tin
2*
Tout
2*+
_gradient_op_typePartitionedCall-4148*R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4142�
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-4181**
config_proto

CPU

GPU 2J 8*
Tout
2*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4175*(
_output_shapes
:����������*
Tin
2�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_4199*
Tin
2*+
_gradient_op_typePartitionedCall-4205*
Tout
2*'
_output_shapes
:���������@�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-4233**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4227*
Tin
2*
Tout
2�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
O
conv2d_input?
serving_default_conv2d_input:0�����������;
dense_10
StatefulPartitionedCall:0���������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�>
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"�;
_tf_keras_sequential�;{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 250, 250, 3], "dtype": "float32", "filters": 96, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": [null, 250, 250, 3], "dtype": "float32", "filters": 256, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": [null, 250, 250, 3], "dtype": "float32", "filters": 512, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 250, 250, 3], "dtype": "float32", "filters": 96, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": [null, 250, 250, 3], "dtype": "float32", "filters": 256, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": [null, 250, 250, 3], "dtype": "float32", "filters": 512, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "conv2d_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 250, 250, 3], "config": {"batch_input_shape": [null, 250, 250, 3], "dtype": "float32", "sparse": false, "name": "conv2d_input"}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 250, 250, 3], "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 250, 250, 3], "dtype": "float32", "filters": 96, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
�
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 250, 250, 3], "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": [null, 250, 250, 3], "dtype": "float32", "filters": 256, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 96}}}}
�
%trainable_variables
&regularization_losses
'	variables
(	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 250, 250, 3], "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": [null, 250, 250, 3], "dtype": "float32", "filters": 512, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
�
/trainable_variables
0regularization_losses
1	variables
2	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
3trainable_variables
4regularization_losses
5	variables
6	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

7kernel
8bias
9trainable_variables
:regularization_losses
;	variables
<	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}}
�

=kernel
>bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
�
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem�m�m� m�)m�*m�7m�8m�=m�>m�v�v�v� v�)v�*v�7v�8v�=v�>v�"
	optimizer
f
0
1
2
 3
)4
*5
76
87
=8
>9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
 3
)4
*5
76
87
=8
>9"
trackable_list_wrapper
�

Hlayers
Ilayer_regularization_losses
trainable_variables
Jnon_trainable_variables
Kmetrics
regularization_losses
	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

Llayers
Mlayer_regularization_losses
trainable_variables
Nnon_trainable_variables
Ometrics
regularization_losses
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%`2conv2d/kernel
:`2conv2d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

Players
Qlayer_regularization_losses
trainable_variables
Rnon_trainable_variables
Smetrics
regularization_losses
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

Tlayers
Ulayer_regularization_losses
trainable_variables
Vnon_trainable_variables
Wmetrics
regularization_losses
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(`�2conv2d_1/kernel
:�2conv2d_1/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
�

Xlayers
Ylayer_regularization_losses
!trainable_variables
Znon_trainable_variables
[metrics
"regularization_losses
#	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

\layers
]layer_regularization_losses
%trainable_variables
^non_trainable_variables
_metrics
&regularization_losses
'	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2conv2d_2/kernel
:�2conv2d_2/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�

`layers
alayer_regularization_losses
+trainable_variables
bnon_trainable_variables
cmetrics
,regularization_losses
-	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

dlayers
elayer_regularization_losses
/trainable_variables
fnon_trainable_variables
gmetrics
0regularization_losses
1	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

hlayers
ilayer_regularization_losses
3trainable_variables
jnon_trainable_variables
kmetrics
4regularization_losses
5	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�@2dense/kernel
:@2
dense/bias
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
�

llayers
mlayer_regularization_losses
9trainable_variables
nnon_trainable_variables
ometrics
:regularization_losses
;	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_1/kernel
:2dense_1/bias
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
�

players
qlayer_regularization_losses
?trainable_variables
rnon_trainable_variables
smetrics
@regularization_losses
A	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
_
0
1
2
3
4
5
6
	7

8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
t0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	utotal
	vcount
w
_fn_kwargs
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
�

|layers
}layer_regularization_losses
xtrainable_variables
~non_trainable_variables
metrics
yregularization_losses
z	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
,:*`2Adam/conv2d/kernel/m
:`2Adam/conv2d/bias/m
/:-`�2Adam/conv2d_1/kernel/m
!:�2Adam/conv2d_1/bias/m
0:.��2Adam/conv2d_2/kernel/m
!:�2Adam/conv2d_2/bias/m
$:"	�@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
%:#@2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
,:*`2Adam/conv2d/kernel/v
:`2Adam/conv2d/bias/v
/:-`�2Adam/conv2d_1/kernel/v
!:�2Adam/conv2d_1/bias/v
0:.��2Adam/conv2d_2/kernel/v
!:�2Adam/conv2d_2/bias/v
$:"	�@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
%:#@2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
�2�
__inference__wrapped_model_4025�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *5�2
0�-
conv2d_input�����������
�2�
)__inference_sequential_layer_call_fn_4351
)__inference_sequential_layer_call_fn_4477
)__inference_sequential_layer_call_fn_4310
)__inference_sequential_layer_call_fn_4492�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_sequential_layer_call_and_return_conditional_losses_4462
D__inference_sequential_layer_call_and_return_conditional_losses_4270
D__inference_sequential_layer_call_and_return_conditional_losses_4418
D__inference_sequential_layer_call_and_return_conditional_losses_4245�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
%__inference_conv2d_layer_call_fn_4050�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
@__inference_conv2d_layer_call_and_return_conditional_losses_4039�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
,__inference_max_pooling2d_layer_call_fn_4067�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4058�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
'__inference_conv2d_1_layer_call_fn_4092�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������`
�2�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4081�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������`
�2�
.__inference_max_pooling2d_1_layer_call_fn_4109�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4100�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
'__inference_conv2d_2_layer_call_fn_4134�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4123�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
.__inference_max_pooling2d_2_layer_call_fn_4151�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4142�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
&__inference_flatten_layer_call_fn_4503�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_flatten_layer_call_and_return_conditional_losses_4498�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_dense_layer_call_fn_4521�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_dense_layer_call_and_return_conditional_losses_4514�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_1_layer_call_fn_4539�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_1_layer_call_and_return_conditional_losses_4532�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
6B4
"__inference_signature_wrapper_4372conv2d_input
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
&__inference_flatten_layer_call_fn_4503U8�5
.�+
)�&
inputs����������
� "������������
)__inference_sequential_layer_call_fn_4351o
 )*78=>G�D
=�:
0�-
conv2d_input�����������
p 

 
� "�����������
D__inference_sequential_layer_call_and_return_conditional_losses_4245|
 )*78=>G�D
=�:
0�-
conv2d_input�����������
p

 
� "%�"
�
0���������
� �
)__inference_sequential_layer_call_fn_4310o
 )*78=>G�D
=�:
0�-
conv2d_input�����������
p

 
� "�����������
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4123�)*J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
@__inference_conv2d_layer_call_and_return_conditional_losses_4039�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������`
� �
)__inference_sequential_layer_call_fn_4492i
 )*78=>A�>
7�4
*�'
inputs�����������
p 

 
� "�����������
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4058�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_max_pooling2d_2_layer_call_fn_4151�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
'__inference_conv2d_2_layer_call_fn_4134�)*J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
D__inference_sequential_layer_call_and_return_conditional_losses_4270|
 )*78=>G�D
=�:
0�-
conv2d_input�����������
p 

 
� "%�"
�
0���������
� �
"__inference_signature_wrapper_4372�
 )*78=>O�L
� 
E�B
@
conv2d_input0�-
conv2d_input�����������"1�.
,
dense_1!�
dense_1����������
A__inference_flatten_layer_call_and_return_conditional_losses_4498b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
'__inference_conv2d_1_layer_call_fn_4092� I�F
?�<
:�7
inputs+���������������������������`
� "3�0,����������������������������x
$__inference_dense_layer_call_fn_4521P780�-
&�#
!�
inputs����������
� "����������@�
D__inference_sequential_layer_call_and_return_conditional_losses_4462v
 )*78=>A�>
7�4
*�'
inputs�����������
p 

 
� "%�"
�
0���������
� �
.__inference_max_pooling2d_1_layer_call_fn_4109�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
A__inference_dense_1_layer_call_and_return_conditional_losses_4532\=>/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� �
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4142�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
,__inference_max_pooling2d_layer_call_fn_4067�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
?__inference_dense_layer_call_and_return_conditional_losses_4514]780�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� �
)__inference_sequential_layer_call_fn_4477i
 )*78=>A�>
7�4
*�'
inputs�����������
p

 
� "�����������
D__inference_sequential_layer_call_and_return_conditional_losses_4418v
 )*78=>A�>
7�4
*�'
inputs�����������
p

 
� "%�"
�
0���������
� �
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4100�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4081� I�F
?�<
:�7
inputs+���������������������������`
� "@�=
6�3
0,����������������������������
� �
%__inference_conv2d_layer_call_fn_4050�I�F
?�<
:�7
inputs+���������������������������
� "2�/+���������������������������`�
__inference__wrapped_model_4025�
 )*78=>?�<
5�2
0�-
conv2d_input�����������
� "1�.
,
dense_1!�
dense_1���������y
&__inference_dense_1_layer_call_fn_4539O=>/�,
%�"
 �
inputs���������@
� "����������