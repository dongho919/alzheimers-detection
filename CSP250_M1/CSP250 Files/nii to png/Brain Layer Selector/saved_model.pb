�� 
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02v2.6.0-rc2-32-g919f693420e8��
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes

:��*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes

:��*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes

:��*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes

:��*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
��*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
�
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
�
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
�
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
�
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*$
shared_nameblock2_conv1/kernel
�
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@�*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:�*
dtype0
�
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock2_conv2/kernel
�
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:��*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:�*
dtype0
�
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock3_conv1/kernel
�
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:��*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:�*
dtype0
�
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock3_conv2/kernel
�
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:��*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:�*
dtype0
�
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock3_conv3/kernel
�
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:��*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:�*
dtype0
�
block3_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock3_conv4/kernel
�
'block3_conv4/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4/kernel*(
_output_shapes
:��*
dtype0
{
block3_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock3_conv4/bias
t
%block3_conv4/bias/Read/ReadVariableOpReadVariableOpblock3_conv4/bias*
_output_shapes	
:�*
dtype0
�
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock4_conv1/kernel
�
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:��*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:�*
dtype0
�
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock4_conv2/kernel
�
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:��*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:�*
dtype0
�
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock4_conv3/kernel
�
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:��*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:�*
dtype0
�
block4_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock4_conv4/kernel
�
'block4_conv4/kernel/Read/ReadVariableOpReadVariableOpblock4_conv4/kernel*(
_output_shapes
:��*
dtype0
{
block4_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock4_conv4/bias
t
%block4_conv4/bias/Read/ReadVariableOpReadVariableOpblock4_conv4/bias*
_output_shapes	
:�*
dtype0
�
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock5_conv1/kernel
�
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:��*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:�*
dtype0
�
block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock5_conv2/kernel
�
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:��*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:�*
dtype0
�
block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock5_conv3/kernel
�
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:��*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:�*
dtype0
�
block5_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameblock5_conv4/kernel
�
'block5_conv4/kernel/Read/ReadVariableOpReadVariableOpblock5_conv4/kernel*(
_output_shapes
:��*
dtype0
{
block5_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameblock5_conv4/bias
t
%block5_conv4/bias/Read/ReadVariableOpReadVariableOpblock5_conv4/bias*
_output_shapes	
:�*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*3
shared_name$"Adam/batch_normalization_1/gamma/m
�
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes

:��*
dtype0
�
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*2
shared_name#!Adam/batch_normalization_1/beta/m
�
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes

:��*
dtype0
�
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/dense_2/kernel/m
�
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m* 
_output_shapes
:
��*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
�
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*3
shared_name$"Adam/batch_normalization_1/gamma/v
�
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes

:��*
dtype0
�
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*2
shared_name#!Adam/batch_normalization_1/beta/v
�
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes

:��*
dtype0
�
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/dense_2/kernel/v
�
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v* 
_output_shapes
:
��*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ۇ
valueЇḂ Bć
�
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	optimizer
trainable_variables
		variables

regularization_losses
	keras_api

signatures
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
 layer_with_weights-14
 layer-19
!layer_with_weights-15
!layer-20
"layer-21
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
R
+trainable_variables
,	variables
-regularization_losses
.	keras_api
�
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
h

>kernel
?bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
�
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_rate0m�1m�8m�9m�>m�?m�0v�1v�8v�9v�>v�?v�
*
00
11
82
93
>4
?5
�
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19
]20
^21
_22
`23
a24
b25
c26
d27
e28
f29
g30
h31
032
133
234
335
836
937
>38
?39
 
�
trainable_variables
inon_trainable_variables
jmetrics
klayer_regularization_losses
		variables

regularization_losses
llayer_metrics

mlayers
 
 
h

Ikernel
Jbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
h

Kkernel
Lbias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
R
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
h

Mkernel
Nbias
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
j

Okernel
Pbias
~trainable_variables
	variables
�regularization_losses
�	keras_api
V
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

Qkernel
Rbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

Skernel
Tbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

Ukernel
Vbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

Wkernel
Xbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
V
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

Ykernel
Zbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

[kernel
\bias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

]kernel
^bias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

_kernel
`bias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
V
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

akernel
bbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

ckernel
dbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

ekernel
fbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
l

gkernel
hbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
V
�trainable_variables
�	variables
�regularization_losses
�	keras_api
 
�
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19
]20
^21
_22
`23
a24
b25
c26
d27
e28
f29
g30
h31
 
�
#trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
$	variables
%regularization_losses
�layer_metrics
�layers
 
 
 
�
'trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
(	variables
)regularization_losses
�layer_metrics
�layers
 
 
 
�
+trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
,	variables
-regularization_losses
�layer_metrics
�layers
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
22
33
 
�
4trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
5	variables
6regularization_losses
�layer_metrics
�layers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
�
:trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
;	variables
<regularization_losses
�layer_metrics
�layers
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1

>0
?1
 
�
@trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
A	variables
Bregularization_losses
�layer_metrics
�layers
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
OM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv2/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv2/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv3/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv3/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv4/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv4/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv1/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv1/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv2/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv2/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv3/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv3/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv4/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv4/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
�
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19
]20
^21
_22
`23
a24
b25
c26
d27
e28
f29
g30
h31
232
333

�0
�1
 
 
*
0
1
2
3
4
5
 

I0
J1
 
�
ntrainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
o	variables
pregularization_losses
�layer_metrics
�layers
 

K0
L1
 
�
rtrainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
s	variables
tregularization_losses
�layer_metrics
�layers
 
 
 
�
vtrainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
w	variables
xregularization_losses
�layer_metrics
�layers
 

M0
N1
 
�
ztrainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
{	variables
|regularization_losses
�layer_metrics
�layers
 

O0
P1
 
�
~trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
	variables
�regularization_losses
�layer_metrics
�layers
 
 
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

Q0
R1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

S0
T1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

U0
V1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

W0
X1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 
 
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

Y0
Z1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

[0
\1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

]0
^1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

_0
`1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 
 
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

a0
b1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

c0
d1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

e0
f1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 

g0
h1
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
 
 
 
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19
]20
^21
_22
`23
a24
b25
c26
d27
e28
f29
g30
h31
 
 
 
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
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

20
31
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
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api

I0
J1
 
 
 
 

K0
L1
 
 
 
 
 
 
 
 
 

M0
N1
 
 
 
 

O0
P1
 
 
 
 
 
 
 
 
 

Q0
R1
 
 
 
 

S0
T1
 
 
 
 

U0
V1
 
 
 
 

W0
X1
 
 
 
 
 
 
 
 
 

Y0
Z1
 
 
 
 

[0
\1
 
 
 
 

]0
^1
 
 
 
 

_0
`1
 
 
 
 
 
 
 
 
 

a0
b1
 
 
 
 

c0
d1
 
 
 
 

e0
f1
 
 
 
 

g0
h1
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
��
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_vgg19_inputPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_vgg19_inputblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_10886
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv4/kernel/Read/ReadVariableOp%block3_conv4/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block4_conv4/kernel/Read/ReadVariableOp%block4_conv4/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp'block5_conv4/kernel/Read/ReadVariableOp%block5_conv4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst*J
TinC
A2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_12549
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biastotalcounttotal_1count_1"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/m"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*I
TinB
@2>*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_12742�
�
�
F__inference_block2_conv2_layer_call_and_return_conditional_losses_8882

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������pp�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������pp�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������pp�
 
_user_specified_nameinputs
�
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_10259

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constu
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:�����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:�����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:�����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:�����������2
dropout/Cast|
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:�����������2
dropout/Mul_1g
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
B__inference_dense_3_layer_call_and_return_conditional_losses_10129

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_block3_conv3_layer_call_and_return_conditional_losses_12094

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
F__inference_block5_conv4_layer_call_and_return_conditional_losses_9104

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_block5_conv1_layer_call_and_return_conditional_losses_12254

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_dropout_1_layer_call_fn_11798

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_100902
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
B__inference_dense_2_layer_call_and_return_conditional_losses_10112

inputs2
matmul_readvariableop_resource:
��-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
G__inference_block3_conv4_layer_call_and_return_conditional_losses_12114

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
G__inference_block2_conv1_layer_call_and_return_conditional_losses_11994

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������pp�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������pp@
 
_user_specified_nameinputs
��
�
@__inference_vgg19_layer_call_and_return_conditional_losses_11506

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@�;
,block2_conv1_biasadd_readvariableop_resource:	�G
+block2_conv2_conv2d_readvariableop_resource:��;
,block2_conv2_biasadd_readvariableop_resource:	�G
+block3_conv1_conv2d_readvariableop_resource:��;
,block3_conv1_biasadd_readvariableop_resource:	�G
+block3_conv2_conv2d_readvariableop_resource:��;
,block3_conv2_biasadd_readvariableop_resource:	�G
+block3_conv3_conv2d_readvariableop_resource:��;
,block3_conv3_biasadd_readvariableop_resource:	�G
+block3_conv4_conv2d_readvariableop_resource:��;
,block3_conv4_biasadd_readvariableop_resource:	�G
+block4_conv1_conv2d_readvariableop_resource:��;
,block4_conv1_biasadd_readvariableop_resource:	�G
+block4_conv2_conv2d_readvariableop_resource:��;
,block4_conv2_biasadd_readvariableop_resource:	�G
+block4_conv3_conv2d_readvariableop_resource:��;
,block4_conv3_biasadd_readvariableop_resource:	�G
+block4_conv4_conv2d_readvariableop_resource:��;
,block4_conv4_biasadd_readvariableop_resource:	�G
+block5_conv1_conv2d_readvariableop_resource:��;
,block5_conv1_biasadd_readvariableop_resource:	�G
+block5_conv2_conv2d_readvariableop_resource:��;
,block5_conv2_biasadd_readvariableop_resource:	�G
+block5_conv3_conv2d_readvariableop_resource:��;
,block5_conv3_biasadd_readvariableop_resource:	�G
+block5_conv4_conv2d_readvariableop_resource:��;
,block5_conv4_biasadd_readvariableop_resource:	�
identity��#block1_conv1/BiasAdd/ReadVariableOp�"block1_conv1/Conv2D/ReadVariableOp�#block1_conv2/BiasAdd/ReadVariableOp�"block1_conv2/Conv2D/ReadVariableOp�#block2_conv1/BiasAdd/ReadVariableOp�"block2_conv1/Conv2D/ReadVariableOp�#block2_conv2/BiasAdd/ReadVariableOp�"block2_conv2/Conv2D/ReadVariableOp�#block3_conv1/BiasAdd/ReadVariableOp�"block3_conv1/Conv2D/ReadVariableOp�#block3_conv2/BiasAdd/ReadVariableOp�"block3_conv2/Conv2D/ReadVariableOp�#block3_conv3/BiasAdd/ReadVariableOp�"block3_conv3/Conv2D/ReadVariableOp�#block3_conv4/BiasAdd/ReadVariableOp�"block3_conv4/Conv2D/ReadVariableOp�#block4_conv1/BiasAdd/ReadVariableOp�"block4_conv1/Conv2D/ReadVariableOp�#block4_conv2/BiasAdd/ReadVariableOp�"block4_conv2/Conv2D/ReadVariableOp�#block4_conv3/BiasAdd/ReadVariableOp�"block4_conv3/Conv2D/ReadVariableOp�#block4_conv4/BiasAdd/ReadVariableOp�"block4_conv4/Conv2D/ReadVariableOp�#block5_conv1/BiasAdd/ReadVariableOp�"block5_conv1/Conv2D/ReadVariableOp�#block5_conv2/BiasAdd/ReadVariableOp�"block5_conv2/Conv2D/ReadVariableOp�#block5_conv3/BiasAdd/ReadVariableOp�"block5_conv3/Conv2D/ReadVariableOp�#block5_conv4/BiasAdd/ReadVariableOp�"block5_conv4/Conv2D/ReadVariableOp�
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp�
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
block1_conv1/Conv2D�
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp�
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2
block1_conv1/BiasAdd�
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
block1_conv1/Relu�
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp�
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
block1_conv2/Conv2D�
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp�
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2
block1_conv2/BiasAdd�
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
block1_conv2/Relu�
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool�
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp�
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
block2_conv1/Conv2D�
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp�
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2
block2_conv1/BiasAdd�
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
block2_conv1/Relu�
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp�
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
block2_conv2/Conv2D�
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp�
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2
block2_conv2/BiasAdd�
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
block2_conv2/Relu�
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:���������88�*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool�
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp�
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
block3_conv1/Conv2D�
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp�
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
block3_conv1/BiasAdd�
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
block3_conv1/Relu�
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp�
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
block3_conv2/Conv2D�
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp�
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
block3_conv2/BiasAdd�
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
block3_conv2/Relu�
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp�
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
block3_conv3/Conv2D�
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp�
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
block3_conv3/BiasAdd�
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
block3_conv3/Relu�
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block3_conv4/Conv2D/ReadVariableOp�
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
block3_conv4/Conv2D�
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block3_conv4/BiasAdd/ReadVariableOp�
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
block3_conv4/BiasAdd�
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
block3_conv4/Relu�
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool�
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp�
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block4_conv1/Conv2D�
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp�
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block4_conv1/BiasAdd�
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block4_conv1/Relu�
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp�
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block4_conv2/Conv2D�
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp�
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block4_conv2/BiasAdd�
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block4_conv2/Relu�
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp�
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block4_conv3/Conv2D�
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp�
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block4_conv3/BiasAdd�
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block4_conv3/Relu�
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block4_conv4/Conv2D/ReadVariableOp�
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block4_conv4/Conv2D�
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block4_conv4/BiasAdd/ReadVariableOp�
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block4_conv4/BiasAdd�
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block4_conv4/Relu�
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool�
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp�
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block5_conv1/Conv2D�
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp�
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block5_conv1/BiasAdd�
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block5_conv1/Relu�
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block5_conv2/Conv2D/ReadVariableOp�
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block5_conv2/Conv2D�
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp�
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block5_conv2/BiasAdd�
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block5_conv2/Relu�
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block5_conv3/Conv2D/ReadVariableOp�
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block5_conv3/Conv2D�
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp�
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block5_conv3/BiasAdd�
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block5_conv3/Relu�
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block5_conv4/Conv2D/ReadVariableOp�
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block5_conv4/Conv2D�
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block5_conv4/BiasAdd/ReadVariableOp�
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block5_conv4/BiasAdd�
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block5_conv4/Relu�
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool�
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identity�	
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
G
+__inference_block3_pool_layer_call_fn_12143

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_89662
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������88�:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_10090

inputs

identity_1\
IdentityIdentityinputs*
T0*)
_output_shapes
:�����������2

Identityk

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:�����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�,
__inference__wrapped_model_8697
vgg19_inputX
>sequential_1_vgg19_block1_conv1_conv2d_readvariableop_resource:@M
?sequential_1_vgg19_block1_conv1_biasadd_readvariableop_resource:@X
>sequential_1_vgg19_block1_conv2_conv2d_readvariableop_resource:@@M
?sequential_1_vgg19_block1_conv2_biasadd_readvariableop_resource:@Y
>sequential_1_vgg19_block2_conv1_conv2d_readvariableop_resource:@�N
?sequential_1_vgg19_block2_conv1_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block2_conv2_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block2_conv2_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block3_conv1_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block3_conv1_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block3_conv2_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block3_conv2_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block3_conv3_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block3_conv3_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block3_conv4_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block3_conv4_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block4_conv1_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block4_conv1_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block4_conv2_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block4_conv2_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block4_conv3_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block4_conv3_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block4_conv4_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block4_conv4_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block5_conv1_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block5_conv1_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block5_conv2_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block5_conv2_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block5_conv3_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block5_conv3_biasadd_readvariableop_resource:	�Z
>sequential_1_vgg19_block5_conv4_conv2d_readvariableop_resource:��N
?sequential_1_vgg19_block5_conv4_biasadd_readvariableop_resource:	�T
Dsequential_1_batch_normalization_1_batchnorm_readvariableop_resource:
��X
Hsequential_1_batch_normalization_1_batchnorm_mul_readvariableop_resource:
��V
Fsequential_1_batch_normalization_1_batchnorm_readvariableop_1_resource:
��V
Fsequential_1_batch_normalization_1_batchnorm_readvariableop_2_resource:
��G
3sequential_1_dense_2_matmul_readvariableop_resource:
��B
4sequential_1_dense_2_biasadd_readvariableop_resource:E
3sequential_1_dense_3_matmul_readvariableop_resource:B
4sequential_1_dense_3_biasadd_readvariableop_resource:
identity��;sequential_1/batch_normalization_1/batchnorm/ReadVariableOp�=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1�=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2�?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp�+sequential_1/dense_2/BiasAdd/ReadVariableOp�*sequential_1/dense_2/MatMul/ReadVariableOp�+sequential_1/dense_3/BiasAdd/ReadVariableOp�*sequential_1/dense_3/MatMul/ReadVariableOp�6sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOp�6sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp�6sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp�6sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp�6sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp�6sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp�6sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp�6sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp�6sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp�6sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp�6sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp�6sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp�6sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp�6sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp�6sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp�6sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp�5sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp�
5sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype027
5sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block1_conv1/Conv2DConv2Dvgg19_input=sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2(
&sequential_1/vgg19/block1_conv1/Conv2D�
6sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block1_conv1/BiasAddBiasAdd/sequential_1/vgg19/block1_conv1/Conv2D:output:0>sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2)
'sequential_1/vgg19/block1_conv1/BiasAdd�
$sequential_1/vgg19/block1_conv1/ReluRelu0sequential_1/vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2&
$sequential_1/vgg19/block1_conv1/Relu�
5sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype027
5sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block1_conv2/Conv2DConv2D2sequential_1/vgg19/block1_conv1/Relu:activations:0=sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2(
&sequential_1/vgg19/block1_conv2/Conv2D�
6sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block1_conv2/BiasAddBiasAdd/sequential_1/vgg19/block1_conv2/Conv2D:output:0>sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2)
'sequential_1/vgg19/block1_conv2/BiasAdd�
$sequential_1/vgg19/block1_conv2/ReluRelu0sequential_1/vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2&
$sequential_1/vgg19/block1_conv2/Relu�
&sequential_1/vgg19/block1_pool/MaxPoolMaxPool2sequential_1/vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
2(
&sequential_1/vgg19/block1_pool/MaxPool�
5sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype027
5sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block2_conv1/Conv2DConv2D/sequential_1/vgg19/block1_pool/MaxPool:output:0=sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2(
&sequential_1/vgg19/block2_conv1/Conv2D�
6sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block2_conv1/BiasAddBiasAdd/sequential_1/vgg19/block2_conv1/Conv2D:output:0>sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2)
'sequential_1/vgg19/block2_conv1/BiasAdd�
$sequential_1/vgg19/block2_conv1/ReluRelu0sequential_1/vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2&
$sequential_1/vgg19/block2_conv1/Relu�
5sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block2_conv2/Conv2DConv2D2sequential_1/vgg19/block2_conv1/Relu:activations:0=sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2(
&sequential_1/vgg19/block2_conv2/Conv2D�
6sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block2_conv2/BiasAddBiasAdd/sequential_1/vgg19/block2_conv2/Conv2D:output:0>sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2)
'sequential_1/vgg19/block2_conv2/BiasAdd�
$sequential_1/vgg19/block2_conv2/ReluRelu0sequential_1/vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2&
$sequential_1/vgg19/block2_conv2/Relu�
&sequential_1/vgg19/block2_pool/MaxPoolMaxPool2sequential_1/vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:���������88�*
ksize
*
paddingVALID*
strides
2(
&sequential_1/vgg19/block2_pool/MaxPool�
5sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block3_conv1/Conv2DConv2D/sequential_1/vgg19/block2_pool/MaxPool:output:0=sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2(
&sequential_1/vgg19/block3_conv1/Conv2D�
6sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block3_conv1/BiasAddBiasAdd/sequential_1/vgg19/block3_conv1/Conv2D:output:0>sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2)
'sequential_1/vgg19/block3_conv1/BiasAdd�
$sequential_1/vgg19/block3_conv1/ReluRelu0sequential_1/vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2&
$sequential_1/vgg19/block3_conv1/Relu�
5sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block3_conv2/Conv2DConv2D2sequential_1/vgg19/block3_conv1/Relu:activations:0=sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2(
&sequential_1/vgg19/block3_conv2/Conv2D�
6sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block3_conv2/BiasAddBiasAdd/sequential_1/vgg19/block3_conv2/Conv2D:output:0>sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2)
'sequential_1/vgg19/block3_conv2/BiasAdd�
$sequential_1/vgg19/block3_conv2/ReluRelu0sequential_1/vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2&
$sequential_1/vgg19/block3_conv2/Relu�
5sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block3_conv3/Conv2DConv2D2sequential_1/vgg19/block3_conv2/Relu:activations:0=sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2(
&sequential_1/vgg19/block3_conv3/Conv2D�
6sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block3_conv3/BiasAddBiasAdd/sequential_1/vgg19/block3_conv3/Conv2D:output:0>sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2)
'sequential_1/vgg19/block3_conv3/BiasAdd�
$sequential_1/vgg19/block3_conv3/ReluRelu0sequential_1/vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2&
$sequential_1/vgg19/block3_conv3/Relu�
5sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block3_conv4/Conv2DConv2D2sequential_1/vgg19/block3_conv3/Relu:activations:0=sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2(
&sequential_1/vgg19/block3_conv4/Conv2D�
6sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block3_conv4/BiasAddBiasAdd/sequential_1/vgg19/block3_conv4/Conv2D:output:0>sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2)
'sequential_1/vgg19/block3_conv4/BiasAdd�
$sequential_1/vgg19/block3_conv4/ReluRelu0sequential_1/vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2&
$sequential_1/vgg19/block3_conv4/Relu�
&sequential_1/vgg19/block3_pool/MaxPoolMaxPool2sequential_1/vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2(
&sequential_1/vgg19/block3_pool/MaxPool�
5sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block4_conv1/Conv2DConv2D/sequential_1/vgg19/block3_pool/MaxPool:output:0=sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&sequential_1/vgg19/block4_conv1/Conv2D�
6sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block4_conv1/BiasAddBiasAdd/sequential_1/vgg19/block4_conv1/Conv2D:output:0>sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'sequential_1/vgg19/block4_conv1/BiasAdd�
$sequential_1/vgg19/block4_conv1/ReluRelu0sequential_1/vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$sequential_1/vgg19/block4_conv1/Relu�
5sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block4_conv2/Conv2DConv2D2sequential_1/vgg19/block4_conv1/Relu:activations:0=sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&sequential_1/vgg19/block4_conv2/Conv2D�
6sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block4_conv2/BiasAddBiasAdd/sequential_1/vgg19/block4_conv2/Conv2D:output:0>sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'sequential_1/vgg19/block4_conv2/BiasAdd�
$sequential_1/vgg19/block4_conv2/ReluRelu0sequential_1/vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$sequential_1/vgg19/block4_conv2/Relu�
5sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block4_conv3/Conv2DConv2D2sequential_1/vgg19/block4_conv2/Relu:activations:0=sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&sequential_1/vgg19/block4_conv3/Conv2D�
6sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block4_conv3/BiasAddBiasAdd/sequential_1/vgg19/block4_conv3/Conv2D:output:0>sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'sequential_1/vgg19/block4_conv3/BiasAdd�
$sequential_1/vgg19/block4_conv3/ReluRelu0sequential_1/vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$sequential_1/vgg19/block4_conv3/Relu�
5sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block4_conv4/Conv2DConv2D2sequential_1/vgg19/block4_conv3/Relu:activations:0=sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&sequential_1/vgg19/block4_conv4/Conv2D�
6sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block4_conv4/BiasAddBiasAdd/sequential_1/vgg19/block4_conv4/Conv2D:output:0>sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'sequential_1/vgg19/block4_conv4/BiasAdd�
$sequential_1/vgg19/block4_conv4/ReluRelu0sequential_1/vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$sequential_1/vgg19/block4_conv4/Relu�
&sequential_1/vgg19/block4_pool/MaxPoolMaxPool2sequential_1/vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2(
&sequential_1/vgg19/block4_pool/MaxPool�
5sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block5_conv1/Conv2DConv2D/sequential_1/vgg19/block4_pool/MaxPool:output:0=sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&sequential_1/vgg19/block5_conv1/Conv2D�
6sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block5_conv1/BiasAddBiasAdd/sequential_1/vgg19/block5_conv1/Conv2D:output:0>sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'sequential_1/vgg19/block5_conv1/BiasAdd�
$sequential_1/vgg19/block5_conv1/ReluRelu0sequential_1/vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$sequential_1/vgg19/block5_conv1/Relu�
5sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block5_conv2/Conv2DConv2D2sequential_1/vgg19/block5_conv1/Relu:activations:0=sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&sequential_1/vgg19/block5_conv2/Conv2D�
6sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block5_conv2/BiasAddBiasAdd/sequential_1/vgg19/block5_conv2/Conv2D:output:0>sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'sequential_1/vgg19/block5_conv2/BiasAdd�
$sequential_1/vgg19/block5_conv2/ReluRelu0sequential_1/vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$sequential_1/vgg19/block5_conv2/Relu�
5sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block5_conv3/Conv2DConv2D2sequential_1/vgg19/block5_conv2/Relu:activations:0=sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&sequential_1/vgg19/block5_conv3/Conv2D�
6sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block5_conv3/BiasAddBiasAdd/sequential_1/vgg19/block5_conv3/Conv2D:output:0>sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'sequential_1/vgg19/block5_conv3/BiasAdd�
$sequential_1/vgg19/block5_conv3/ReluRelu0sequential_1/vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$sequential_1/vgg19/block5_conv3/Relu�
5sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype027
5sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp�
&sequential_1/vgg19/block5_conv4/Conv2DConv2D2sequential_1/vgg19/block5_conv3/Relu:activations:0=sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2(
&sequential_1/vgg19/block5_conv4/Conv2D�
6sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp�
'sequential_1/vgg19/block5_conv4/BiasAddBiasAdd/sequential_1/vgg19/block5_conv4/Conv2D:output:0>sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2)
'sequential_1/vgg19/block5_conv4/BiasAdd�
$sequential_1/vgg19/block5_conv4/ReluRelu0sequential_1/vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2&
$sequential_1/vgg19/block5_conv4/Relu�
&sequential_1/vgg19/block5_pool/MaxPoolMaxPool2sequential_1/vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2(
&sequential_1/vgg19/block5_pool/MaxPool�
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  2
sequential_1/flatten_1/Const�
sequential_1/flatten_1/ReshapeReshape/sequential_1/vgg19/block5_pool/MaxPool:output:0%sequential_1/flatten_1/Const:output:0*
T0*)
_output_shapes
:�����������2 
sequential_1/flatten_1/Reshape�
sequential_1/dropout_1/IdentityIdentity'sequential_1/flatten_1/Reshape:output:0*
T0*)
_output_shapes
:�����������2!
sequential_1/dropout_1/Identity�
;sequential_1/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpDsequential_1_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes

:��*
dtype02=
;sequential_1/batch_normalization_1/batchnorm/ReadVariableOp�
2sequential_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:24
2sequential_1/batch_normalization_1/batchnorm/add/y�
0sequential_1/batch_normalization_1/batchnorm/addAddV2Csequential_1/batch_normalization_1/batchnorm/ReadVariableOp:value:0;sequential_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes

:��22
0sequential_1/batch_normalization_1/batchnorm/add�
2sequential_1/batch_normalization_1/batchnorm/RsqrtRsqrt4sequential_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes

:��24
2sequential_1/batch_normalization_1/batchnorm/Rsqrt�
?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_1_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes

:��*
dtype02A
?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp�
0sequential_1/batch_normalization_1/batchnorm/mulMul6sequential_1/batch_normalization_1/batchnorm/Rsqrt:y:0Gsequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:��22
0sequential_1/batch_normalization_1/batchnorm/mul�
2sequential_1/batch_normalization_1/batchnorm/mul_1Mul(sequential_1/dropout_1/Identity:output:04sequential_1/batch_normalization_1/batchnorm/mul:z:0*
T0*)
_output_shapes
:�����������24
2sequential_1/batch_normalization_1/batchnorm/mul_1�
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_1_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes

:��*
dtype02?
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1�
2sequential_1/batch_normalization_1/batchnorm/mul_2MulEsequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1:value:04sequential_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes

:��24
2sequential_1/batch_normalization_1/batchnorm/mul_2�
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_1_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes

:��*
dtype02?
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2�
0sequential_1/batch_normalization_1/batchnorm/subSubEsequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2:value:06sequential_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes

:��22
0sequential_1/batch_normalization_1/batchnorm/sub�
2sequential_1/batch_normalization_1/batchnorm/add_1AddV26sequential_1/batch_normalization_1/batchnorm/mul_1:z:04sequential_1/batch_normalization_1/batchnorm/sub:z:0*
T0*)
_output_shapes
:�����������24
2sequential_1/batch_normalization_1/batchnorm/add_1�
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02,
*sequential_1/dense_2/MatMul/ReadVariableOp�
sequential_1/dense_2/MatMulMatMul6sequential_1/batch_normalization_1/batchnorm/add_1:z:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_1/dense_2/MatMul�
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp�
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_1/dense_2/BiasAdd�
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential_1/dense_2/Relu�
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOp�
sequential_1/dense_3/MatMulMatMul'sequential_1/dense_2/Relu:activations:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_1/dense_3/MatMul�
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp�
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_1/dense_3/BiasAdd�
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential_1/dense_3/Softmax�
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp<^sequential_1/batch_normalization_1/batchnorm/ReadVariableOp>^sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1>^sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2@^sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp7^sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp7^sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp7^sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp7^sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp7^sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp7^sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp6^sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2z
;sequential_1/batch_normalization_1/batchnorm/ReadVariableOp;sequential_1/batch_normalization_1/batchnorm/ReadVariableOp2~
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_12~
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_22�
?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2p
6sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOp5sequential_1/vgg19/block1_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp5sequential_1/vgg19/block1_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp5sequential_1/vgg19/block2_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp5sequential_1/vgg19/block2_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp5sequential_1/vgg19/block3_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp5sequential_1/vgg19/block3_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp6sequential_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp5sequential_1/vgg19/block3_conv3/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp6sequential_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp5sequential_1/vgg19/block3_conv4/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp5sequential_1/vgg19/block4_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp5sequential_1/vgg19/block4_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp6sequential_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp5sequential_1/vgg19/block4_conv3/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp6sequential_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp5sequential_1/vgg19/block4_conv4/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp5sequential_1/vgg19/block5_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp5sequential_1/vgg19/block5_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp6sequential_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp5sequential_1/vgg19/block5_conv3/Conv2D/ReadVariableOp2p
6sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp6sequential_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp2n
5sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp5sequential_1/vgg19/block5_conv4/Conv2D/ReadVariableOp:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg19_input
�
�
,__inference_block5_conv1_layer_call_fn_12263

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_90532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_block1_conv2_layer_call_and_return_conditional_losses_11954

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
G
+__inference_block5_pool_layer_call_fn_12343

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_91142
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_12028

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_block1_conv2_layer_call_fn_11963

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_88422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_12133

inputs
identity�
MaxPoolMaxPoolinputs*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������88�:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
%__inference_vgg19_layer_call_fn_11765

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vgg19_layer_call_and_return_conditional_losses_95292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
,__inference_block4_conv4_layer_call_fn_12223

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_90302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_block5_conv3_layer_call_and_return_conditional_losses_12294

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_block5_conv3_layer_call_fn_12303

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_90872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�

#__inference_signature_wrapper_10886
vgg19_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�

unknown_31:
��

unknown_32:
��

unknown_33:
��

unknown_34:
��

unknown_35:
��

unknown_36:

unknown_37:

unknown_38:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvgg19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__wrapped_model_86972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg19_input
�
�
B__inference_dense_2_layer_call_and_return_conditional_losses_11894

inputs2
matmul_readvariableop_resource:
��-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�+
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9927

inputs7
'assignmovingavg_readvariableop_resource:
��9
)assignmovingavg_1_readvariableop_resource:
��5
%batchnorm_mul_readvariableop_resource:
��1
!batchnorm_readvariableop_resource:
��
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0* 
_output_shapes
:
��*
	keep_dims(2
moments/mean~
moments/StopGradientStopGradientmoments/mean:output:0*
T0* 
_output_shapes
:
��2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*)
_output_shapes
:�����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0* 
_output_shapes
:
��*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes

:��*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes

:��*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes

:��*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes

:��2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes

:��2
AssignMovingAvg/mul�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes

:��*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes

:��2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes

:��2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes

:��2
batchnorm/adde
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes

:��2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes

:��*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:��2
batchnorm/mulx
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*)
_output_shapes
:�����������2
batchnorm/mul_1}
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes

:��2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes

:��*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes

:��2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*)
_output_shapes
:�����������2
batchnorm/add_1p
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*)
_output_shapes
:�����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�

,__inference_sequential_1_layer_call_fn_10613
vgg19_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�

unknown_31:
��

unknown_32:
��

unknown_33:
��

unknown_34:
��

unknown_35:
��

unknown_36:

unknown_37:

unknown_38:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvgg19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(&	
 #$%&'(*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_104452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg19_input
�
�
G__inference_block4_conv2_layer_call_and_return_conditional_losses_12174

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_block1_pool_layer_call_fn_11983

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_88522
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������pp@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_12328

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
ӧ
�%
G__inference_sequential_1_layer_call_and_return_conditional_losses_11215

inputsK
1vgg19_block1_conv1_conv2d_readvariableop_resource:@@
2vgg19_block1_conv1_biasadd_readvariableop_resource:@K
1vgg19_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg19_block1_conv2_biasadd_readvariableop_resource:@L
1vgg19_block2_conv1_conv2d_readvariableop_resource:@�A
2vgg19_block2_conv1_biasadd_readvariableop_resource:	�M
1vgg19_block2_conv2_conv2d_readvariableop_resource:��A
2vgg19_block2_conv2_biasadd_readvariableop_resource:	�M
1vgg19_block3_conv1_conv2d_readvariableop_resource:��A
2vgg19_block3_conv1_biasadd_readvariableop_resource:	�M
1vgg19_block3_conv2_conv2d_readvariableop_resource:��A
2vgg19_block3_conv2_biasadd_readvariableop_resource:	�M
1vgg19_block3_conv3_conv2d_readvariableop_resource:��A
2vgg19_block3_conv3_biasadd_readvariableop_resource:	�M
1vgg19_block3_conv4_conv2d_readvariableop_resource:��A
2vgg19_block3_conv4_biasadd_readvariableop_resource:	�M
1vgg19_block4_conv1_conv2d_readvariableop_resource:��A
2vgg19_block4_conv1_biasadd_readvariableop_resource:	�M
1vgg19_block4_conv2_conv2d_readvariableop_resource:��A
2vgg19_block4_conv2_biasadd_readvariableop_resource:	�M
1vgg19_block4_conv3_conv2d_readvariableop_resource:��A
2vgg19_block4_conv3_biasadd_readvariableop_resource:	�M
1vgg19_block4_conv4_conv2d_readvariableop_resource:��A
2vgg19_block4_conv4_biasadd_readvariableop_resource:	�M
1vgg19_block5_conv1_conv2d_readvariableop_resource:��A
2vgg19_block5_conv1_biasadd_readvariableop_resource:	�M
1vgg19_block5_conv2_conv2d_readvariableop_resource:��A
2vgg19_block5_conv2_biasadd_readvariableop_resource:	�M
1vgg19_block5_conv3_conv2d_readvariableop_resource:��A
2vgg19_block5_conv3_biasadd_readvariableop_resource:	�M
1vgg19_block5_conv4_conv2d_readvariableop_resource:��A
2vgg19_block5_conv4_biasadd_readvariableop_resource:	�M
=batch_normalization_1_assignmovingavg_readvariableop_resource:
��O
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:
��K
;batch_normalization_1_batchnorm_mul_readvariableop_resource:
��G
7batch_normalization_1_batchnorm_readvariableop_resource:
��:
&dense_2_matmul_readvariableop_resource:
��5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity��%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�)vgg19/block1_conv1/BiasAdd/ReadVariableOp�(vgg19/block1_conv1/Conv2D/ReadVariableOp�)vgg19/block1_conv2/BiasAdd/ReadVariableOp�(vgg19/block1_conv2/Conv2D/ReadVariableOp�)vgg19/block2_conv1/BiasAdd/ReadVariableOp�(vgg19/block2_conv1/Conv2D/ReadVariableOp�)vgg19/block2_conv2/BiasAdd/ReadVariableOp�(vgg19/block2_conv2/Conv2D/ReadVariableOp�)vgg19/block3_conv1/BiasAdd/ReadVariableOp�(vgg19/block3_conv1/Conv2D/ReadVariableOp�)vgg19/block3_conv2/BiasAdd/ReadVariableOp�(vgg19/block3_conv2/Conv2D/ReadVariableOp�)vgg19/block3_conv3/BiasAdd/ReadVariableOp�(vgg19/block3_conv3/Conv2D/ReadVariableOp�)vgg19/block3_conv4/BiasAdd/ReadVariableOp�(vgg19/block3_conv4/Conv2D/ReadVariableOp�)vgg19/block4_conv1/BiasAdd/ReadVariableOp�(vgg19/block4_conv1/Conv2D/ReadVariableOp�)vgg19/block4_conv2/BiasAdd/ReadVariableOp�(vgg19/block4_conv2/Conv2D/ReadVariableOp�)vgg19/block4_conv3/BiasAdd/ReadVariableOp�(vgg19/block4_conv3/Conv2D/ReadVariableOp�)vgg19/block4_conv4/BiasAdd/ReadVariableOp�(vgg19/block4_conv4/Conv2D/ReadVariableOp�)vgg19/block5_conv1/BiasAdd/ReadVariableOp�(vgg19/block5_conv1/Conv2D/ReadVariableOp�)vgg19/block5_conv2/BiasAdd/ReadVariableOp�(vgg19/block5_conv2/Conv2D/ReadVariableOp�)vgg19/block5_conv3/BiasAdd/ReadVariableOp�(vgg19/block5_conv3/Conv2D/ReadVariableOp�)vgg19/block5_conv4/BiasAdd/ReadVariableOp�(vgg19/block5_conv4/Conv2D/ReadVariableOp�
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(vgg19/block1_conv1/Conv2D/ReadVariableOp�
vgg19/block1_conv1/Conv2DConv2Dinputs0vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
vgg19/block1_conv1/Conv2D�
)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg19/block1_conv1/BiasAdd/ReadVariableOp�
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2
vgg19/block1_conv1/BiasAdd�
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
vgg19/block1_conv1/Relu�
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(vgg19/block1_conv2/Conv2D/ReadVariableOp�
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
vgg19/block1_conv2/Conv2D�
)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg19/block1_conv2/BiasAdd/ReadVariableOp�
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2
vgg19/block1_conv2/BiasAdd�
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
vgg19/block1_conv2/Relu�
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
2
vgg19/block1_pool/MaxPool�
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02*
(vgg19/block2_conv1/Conv2D/ReadVariableOp�
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
vgg19/block2_conv1/Conv2D�
)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block2_conv1/BiasAdd/ReadVariableOp�
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2
vgg19/block2_conv1/BiasAdd�
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
vgg19/block2_conv1/Relu�
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block2_conv2/Conv2D/ReadVariableOp�
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
vgg19/block2_conv2/Conv2D�
)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block2_conv2/BiasAdd/ReadVariableOp�
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2
vgg19/block2_conv2/BiasAdd�
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
vgg19/block2_conv2/Relu�
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:���������88�*
ksize
*
paddingVALID*
strides
2
vgg19/block2_pool/MaxPool�
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block3_conv1/Conv2D/ReadVariableOp�
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
vgg19/block3_conv1/Conv2D�
)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block3_conv1/BiasAdd/ReadVariableOp�
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv1/BiasAdd�
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv1/Relu�
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block3_conv2/Conv2D/ReadVariableOp�
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
vgg19/block3_conv2/Conv2D�
)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block3_conv2/BiasAdd/ReadVariableOp�
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv2/BiasAdd�
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv2/Relu�
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block3_conv3/Conv2D/ReadVariableOp�
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
vgg19/block3_conv3/Conv2D�
)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block3_conv3/BiasAdd/ReadVariableOp�
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv3/BiasAdd�
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv3/Relu�
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block3_conv4/Conv2D/ReadVariableOp�
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
vgg19/block3_conv4/Conv2D�
)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block3_conv4/BiasAdd/ReadVariableOp�
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv4/BiasAdd�
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv4/Relu�
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
vgg19/block3_pool/MaxPool�
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block4_conv1/Conv2D/ReadVariableOp�
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block4_conv1/Conv2D�
)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block4_conv1/BiasAdd/ReadVariableOp�
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv1/BiasAdd�
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv1/Relu�
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block4_conv2/Conv2D/ReadVariableOp�
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block4_conv2/Conv2D�
)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block4_conv2/BiasAdd/ReadVariableOp�
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv2/BiasAdd�
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv2/Relu�
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block4_conv3/Conv2D/ReadVariableOp�
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block4_conv3/Conv2D�
)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block4_conv3/BiasAdd/ReadVariableOp�
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv3/BiasAdd�
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv3/Relu�
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block4_conv4/Conv2D/ReadVariableOp�
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block4_conv4/Conv2D�
)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block4_conv4/BiasAdd/ReadVariableOp�
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv4/BiasAdd�
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv4/Relu�
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
vgg19/block4_pool/MaxPool�
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block5_conv1/Conv2D/ReadVariableOp�
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block5_conv1/Conv2D�
)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block5_conv1/BiasAdd/ReadVariableOp�
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv1/BiasAdd�
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv1/Relu�
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block5_conv2/Conv2D/ReadVariableOp�
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block5_conv2/Conv2D�
)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block5_conv2/BiasAdd/ReadVariableOp�
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv2/BiasAdd�
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv2/Relu�
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block5_conv3/Conv2D/ReadVariableOp�
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block5_conv3/Conv2D�
)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block5_conv3/BiasAdd/ReadVariableOp�
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv3/BiasAdd�
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv3/Relu�
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block5_conv4/Conv2D/ReadVariableOp�
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block5_conv4/Conv2D�
)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block5_conv4/BiasAdd/ReadVariableOp�
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv4/BiasAdd�
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv4/Relu�
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
vgg19/block5_pool/MaxPools
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  2
flatten_1/Const�
flatten_1/ReshapeReshape"vgg19/block5_pool/MaxPool:output:0flatten_1/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_1/Reshapew
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const�
dropout_1/dropout/MulMulflatten_1/Reshape:output:0 dropout_1/dropout/Const:output:0*
T0*)
_output_shapes
:�����������2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapeflatten_1/Reshape:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*)
_output_shapes
:�����������*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform�
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:�����������2 
dropout_1/dropout/GreaterEqual�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:�����������2
dropout_1/dropout/Cast�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*)
_output_shapes
:�����������2
dropout_1/dropout/Mul_1�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indices�
"batch_normalization_1/moments/meanMeandropout_1/dropout/Mul_1:z:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0* 
_output_shapes
:
��*
	keep_dims(2$
"batch_normalization_1/moments/mean�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0* 
_output_shapes
:
��2,
*batch_normalization_1/moments/StopGradient�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedropout_1/dropout/Mul_1:z:03batch_normalization_1/moments/StopGradient:output:0*
T0*)
_output_shapes
:�����������21
/batch_normalization_1/moments/SquaredDifference�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices�
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0* 
_output_shapes
:
��*
	keep_dims(2(
&batch_normalization_1/moments/variance�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:��*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze�
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes

:��*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1�
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_1/AssignMovingAvg/decay�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes

:��*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes

:��2+
)batch_normalization_1/AssignMovingAvg/sub�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes

:��2+
)batch_normalization_1/AssignMovingAvg/mul�
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg�
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/AssignMovingAvg_1/decay�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes

:��*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes

:��2-
+batch_normalization_1/AssignMovingAvg_1/sub�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes

:��2-
+batch_normalization_1/AssignMovingAvg_1/mul�
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1�
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_1/batchnorm/add/y�
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes

:��2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes

:��2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes

:��*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:��2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldropout_1/dropout/Mul_1:z:0'batch_normalization_1/batchnorm/mul:z:0*
T0*)
_output_shapes
:�����������2'
%batch_normalization_1/batchnorm/mul_1�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes

:��2'
%batch_normalization_1/batchnorm/mul_2�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes

:��*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes

:��2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*)
_output_shapes
:�����������2'
%batch_normalization_1/batchnorm/add_1�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_2/Relu�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_3/Softmaxt
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�l
�
?__inference_vgg19_layer_call_and_return_conditional_losses_9843
input_2+
block1_conv1_9757:@
block1_conv1_9759:@+
block1_conv2_9762:@@
block1_conv2_9764:@,
block2_conv1_9768:@� 
block2_conv1_9770:	�-
block2_conv2_9773:�� 
block2_conv2_9775:	�-
block3_conv1_9779:�� 
block3_conv1_9781:	�-
block3_conv2_9784:�� 
block3_conv2_9786:	�-
block3_conv3_9789:�� 
block3_conv3_9791:	�-
block3_conv4_9794:�� 
block3_conv4_9796:	�-
block4_conv1_9800:�� 
block4_conv1_9802:	�-
block4_conv2_9805:�� 
block4_conv2_9807:	�-
block4_conv3_9810:�� 
block4_conv3_9812:	�-
block4_conv4_9815:�� 
block4_conv4_9817:	�-
block5_conv1_9821:�� 
block5_conv1_9823:	�-
block5_conv2_9826:�� 
block5_conv2_9828:	�-
block5_conv3_9831:�� 
block5_conv3_9833:	�-
block5_conv4_9836:�� 
block5_conv4_9838:	�
identity��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�$block2_conv1/StatefulPartitionedCall�$block2_conv2/StatefulPartitionedCall�$block3_conv1/StatefulPartitionedCall�$block3_conv2/StatefulPartitionedCall�$block3_conv3/StatefulPartitionedCall�$block3_conv4/StatefulPartitionedCall�$block4_conv1/StatefulPartitionedCall�$block4_conv2/StatefulPartitionedCall�$block4_conv3/StatefulPartitionedCall�$block4_conv4/StatefulPartitionedCall�$block5_conv1/StatefulPartitionedCall�$block5_conv2/StatefulPartitionedCall�$block5_conv3/StatefulPartitionedCall�$block5_conv4/StatefulPartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_9757block1_conv1_9759*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_88252&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_9762block1_conv2_9764*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_88422&
$block1_conv2/StatefulPartitionedCall�
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_88522
block1_pool/PartitionedCall�
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_9768block2_conv1_9770*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_88652&
$block2_conv1/StatefulPartitionedCall�
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_9773block2_conv2_9775*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_88822&
$block2_conv2/StatefulPartitionedCall�
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_88922
block2_pool/PartitionedCall�
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_9779block3_conv1_9781*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_89052&
$block3_conv1/StatefulPartitionedCall�
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_9784block3_conv2_9786*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_89222&
$block3_conv2/StatefulPartitionedCall�
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_9789block3_conv3_9791*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_89392&
$block3_conv3/StatefulPartitionedCall�
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_9794block3_conv4_9796*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_89562&
$block3_conv4/StatefulPartitionedCall�
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_89662
block3_pool/PartitionedCall�
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_9800block4_conv1_9802*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_89792&
$block4_conv1/StatefulPartitionedCall�
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_9805block4_conv2_9807*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_89962&
$block4_conv2/StatefulPartitionedCall�
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_9810block4_conv3_9812*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_90132&
$block4_conv3/StatefulPartitionedCall�
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_9815block4_conv4_9817*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_90302&
$block4_conv4/StatefulPartitionedCall�
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_90402
block4_pool/PartitionedCall�
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_9821block5_conv1_9823*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_90532&
$block5_conv1/StatefulPartitionedCall�
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_9826block5_conv2_9828*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_90702&
$block5_conv2/StatefulPartitionedCall�
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_9831block5_conv3_9833*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_90872&
$block5_conv3/StatefulPartitionedCall�
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_9836block5_conv4_9838*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_91042&
$block5_conv4/StatefulPartitionedCall�
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_91142
block5_pool/PartitionedCall�
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identity�
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_12333

inputs
identity�
MaxPoolMaxPoolinputs*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_11968

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�$
G__inference_sequential_1_layer_call_and_return_conditional_losses_11040

inputsK
1vgg19_block1_conv1_conv2d_readvariableop_resource:@@
2vgg19_block1_conv1_biasadd_readvariableop_resource:@K
1vgg19_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg19_block1_conv2_biasadd_readvariableop_resource:@L
1vgg19_block2_conv1_conv2d_readvariableop_resource:@�A
2vgg19_block2_conv1_biasadd_readvariableop_resource:	�M
1vgg19_block2_conv2_conv2d_readvariableop_resource:��A
2vgg19_block2_conv2_biasadd_readvariableop_resource:	�M
1vgg19_block3_conv1_conv2d_readvariableop_resource:��A
2vgg19_block3_conv1_biasadd_readvariableop_resource:	�M
1vgg19_block3_conv2_conv2d_readvariableop_resource:��A
2vgg19_block3_conv2_biasadd_readvariableop_resource:	�M
1vgg19_block3_conv3_conv2d_readvariableop_resource:��A
2vgg19_block3_conv3_biasadd_readvariableop_resource:	�M
1vgg19_block3_conv4_conv2d_readvariableop_resource:��A
2vgg19_block3_conv4_biasadd_readvariableop_resource:	�M
1vgg19_block4_conv1_conv2d_readvariableop_resource:��A
2vgg19_block4_conv1_biasadd_readvariableop_resource:	�M
1vgg19_block4_conv2_conv2d_readvariableop_resource:��A
2vgg19_block4_conv2_biasadd_readvariableop_resource:	�M
1vgg19_block4_conv3_conv2d_readvariableop_resource:��A
2vgg19_block4_conv3_biasadd_readvariableop_resource:	�M
1vgg19_block4_conv4_conv2d_readvariableop_resource:��A
2vgg19_block4_conv4_biasadd_readvariableop_resource:	�M
1vgg19_block5_conv1_conv2d_readvariableop_resource:��A
2vgg19_block5_conv1_biasadd_readvariableop_resource:	�M
1vgg19_block5_conv2_conv2d_readvariableop_resource:��A
2vgg19_block5_conv2_biasadd_readvariableop_resource:	�M
1vgg19_block5_conv3_conv2d_readvariableop_resource:��A
2vgg19_block5_conv3_biasadd_readvariableop_resource:	�M
1vgg19_block5_conv4_conv2d_readvariableop_resource:��A
2vgg19_block5_conv4_biasadd_readvariableop_resource:	�G
7batch_normalization_1_batchnorm_readvariableop_resource:
��K
;batch_normalization_1_batchnorm_mul_readvariableop_resource:
��I
9batch_normalization_1_batchnorm_readvariableop_1_resource:
��I
9batch_normalization_1_batchnorm_readvariableop_2_resource:
��:
&dense_2_matmul_readvariableop_resource:
��5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity��.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�)vgg19/block1_conv1/BiasAdd/ReadVariableOp�(vgg19/block1_conv1/Conv2D/ReadVariableOp�)vgg19/block1_conv2/BiasAdd/ReadVariableOp�(vgg19/block1_conv2/Conv2D/ReadVariableOp�)vgg19/block2_conv1/BiasAdd/ReadVariableOp�(vgg19/block2_conv1/Conv2D/ReadVariableOp�)vgg19/block2_conv2/BiasAdd/ReadVariableOp�(vgg19/block2_conv2/Conv2D/ReadVariableOp�)vgg19/block3_conv1/BiasAdd/ReadVariableOp�(vgg19/block3_conv1/Conv2D/ReadVariableOp�)vgg19/block3_conv2/BiasAdd/ReadVariableOp�(vgg19/block3_conv2/Conv2D/ReadVariableOp�)vgg19/block3_conv3/BiasAdd/ReadVariableOp�(vgg19/block3_conv3/Conv2D/ReadVariableOp�)vgg19/block3_conv4/BiasAdd/ReadVariableOp�(vgg19/block3_conv4/Conv2D/ReadVariableOp�)vgg19/block4_conv1/BiasAdd/ReadVariableOp�(vgg19/block4_conv1/Conv2D/ReadVariableOp�)vgg19/block4_conv2/BiasAdd/ReadVariableOp�(vgg19/block4_conv2/Conv2D/ReadVariableOp�)vgg19/block4_conv3/BiasAdd/ReadVariableOp�(vgg19/block4_conv3/Conv2D/ReadVariableOp�)vgg19/block4_conv4/BiasAdd/ReadVariableOp�(vgg19/block4_conv4/Conv2D/ReadVariableOp�)vgg19/block5_conv1/BiasAdd/ReadVariableOp�(vgg19/block5_conv1/Conv2D/ReadVariableOp�)vgg19/block5_conv2/BiasAdd/ReadVariableOp�(vgg19/block5_conv2/Conv2D/ReadVariableOp�)vgg19/block5_conv3/BiasAdd/ReadVariableOp�(vgg19/block5_conv3/Conv2D/ReadVariableOp�)vgg19/block5_conv4/BiasAdd/ReadVariableOp�(vgg19/block5_conv4/Conv2D/ReadVariableOp�
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(vgg19/block1_conv1/Conv2D/ReadVariableOp�
vgg19/block1_conv1/Conv2DConv2Dinputs0vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
vgg19/block1_conv1/Conv2D�
)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg19/block1_conv1/BiasAdd/ReadVariableOp�
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2
vgg19/block1_conv1/BiasAdd�
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
vgg19/block1_conv1/Relu�
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(vgg19/block1_conv2/Conv2D/ReadVariableOp�
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
vgg19/block1_conv2/Conv2D�
)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)vgg19/block1_conv2/BiasAdd/ReadVariableOp�
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2
vgg19/block1_conv2/BiasAdd�
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
vgg19/block1_conv2/Relu�
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
2
vgg19/block1_pool/MaxPool�
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02*
(vgg19/block2_conv1/Conv2D/ReadVariableOp�
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
vgg19/block2_conv1/Conv2D�
)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block2_conv1/BiasAdd/ReadVariableOp�
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2
vgg19/block2_conv1/BiasAdd�
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
vgg19/block2_conv1/Relu�
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block2_conv2/Conv2D/ReadVariableOp�
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
vgg19/block2_conv2/Conv2D�
)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block2_conv2/BiasAdd/ReadVariableOp�
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2
vgg19/block2_conv2/BiasAdd�
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
vgg19/block2_conv2/Relu�
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:���������88�*
ksize
*
paddingVALID*
strides
2
vgg19/block2_pool/MaxPool�
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block3_conv1/Conv2D/ReadVariableOp�
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
vgg19/block3_conv1/Conv2D�
)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block3_conv1/BiasAdd/ReadVariableOp�
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv1/BiasAdd�
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv1/Relu�
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block3_conv2/Conv2D/ReadVariableOp�
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
vgg19/block3_conv2/Conv2D�
)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block3_conv2/BiasAdd/ReadVariableOp�
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv2/BiasAdd�
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv2/Relu�
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block3_conv3/Conv2D/ReadVariableOp�
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
vgg19/block3_conv3/Conv2D�
)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block3_conv3/BiasAdd/ReadVariableOp�
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv3/BiasAdd�
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv3/Relu�
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block3_conv4/Conv2D/ReadVariableOp�
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
vgg19/block3_conv4/Conv2D�
)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block3_conv4/BiasAdd/ReadVariableOp�
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv4/BiasAdd�
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
vgg19/block3_conv4/Relu�
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
vgg19/block3_pool/MaxPool�
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block4_conv1/Conv2D/ReadVariableOp�
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block4_conv1/Conv2D�
)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block4_conv1/BiasAdd/ReadVariableOp�
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv1/BiasAdd�
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv1/Relu�
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block4_conv2/Conv2D/ReadVariableOp�
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block4_conv2/Conv2D�
)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block4_conv2/BiasAdd/ReadVariableOp�
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv2/BiasAdd�
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv2/Relu�
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block4_conv3/Conv2D/ReadVariableOp�
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block4_conv3/Conv2D�
)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block4_conv3/BiasAdd/ReadVariableOp�
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv3/BiasAdd�
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv3/Relu�
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block4_conv4/Conv2D/ReadVariableOp�
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block4_conv4/Conv2D�
)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block4_conv4/BiasAdd/ReadVariableOp�
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv4/BiasAdd�
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block4_conv4/Relu�
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
vgg19/block4_pool/MaxPool�
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block5_conv1/Conv2D/ReadVariableOp�
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block5_conv1/Conv2D�
)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block5_conv1/BiasAdd/ReadVariableOp�
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv1/BiasAdd�
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv1/Relu�
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block5_conv2/Conv2D/ReadVariableOp�
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block5_conv2/Conv2D�
)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block5_conv2/BiasAdd/ReadVariableOp�
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv2/BiasAdd�
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv2/Relu�
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block5_conv3/Conv2D/ReadVariableOp�
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block5_conv3/Conv2D�
)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block5_conv3/BiasAdd/ReadVariableOp�
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv3/BiasAdd�
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv3/Relu�
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(vgg19/block5_conv4/Conv2D/ReadVariableOp�
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
vgg19/block5_conv4/Conv2D�
)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)vgg19/block5_conv4/BiasAdd/ReadVariableOp�
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv4/BiasAdd�
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
vgg19/block5_conv4/Relu�
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
vgg19/block5_pool/MaxPools
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  2
flatten_1/Const�
flatten_1/ReshapeReshape"vgg19/block5_pool/MaxPool:output:0flatten_1/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten_1/Reshape�
dropout_1/IdentityIdentityflatten_1/Reshape:output:0*
T0*)
_output_shapes
:�����������2
dropout_1/Identity�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes

:��*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp�
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_1/batchnorm/add/y�
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes

:��2%
#batch_normalization_1/batchnorm/add�
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes

:��2'
%batch_normalization_1/batchnorm/Rsqrt�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes

:��*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:��2%
#batch_normalization_1/batchnorm/mul�
%batch_normalization_1/batchnorm/mul_1Muldropout_1/Identity:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*)
_output_shapes
:�����������2'
%batch_normalization_1/batchnorm/mul_1�
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes

:��*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes

:��2'
%batch_normalization_1/batchnorm/mul_2�
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes

:��*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes

:��2%
#batch_normalization_1/batchnorm/sub�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*)
_output_shapes
:�����������2'
%batch_normalization_1/batchnorm/add_1�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_2/Relu�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_3/Softmaxt
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_12228

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
G__inference_block5_conv2_layer_call_and_return_conditional_losses_12274

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_block2_pool_layer_call_fn_12043

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_88922
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������88�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������pp�:X T
0
_output_shapes
:���������pp�
 
_user_specified_nameinputs
�
�
F__inference_block3_conv1_layer_call_and_return_conditional_losses_8905

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
$__inference_vgg19_layer_call_fn_9184
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vgg19_layer_call_and_return_conditional_losses_91172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
�
F__inference_block2_conv1_layer_call_and_return_conditional_losses_8865

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������pp�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������pp@
 
_user_specified_nameinputs
�l
�
?__inference_vgg19_layer_call_and_return_conditional_losses_9117

inputs+
block1_conv1_8826:@
block1_conv1_8828:@+
block1_conv2_8843:@@
block1_conv2_8845:@,
block2_conv1_8866:@� 
block2_conv1_8868:	�-
block2_conv2_8883:�� 
block2_conv2_8885:	�-
block3_conv1_8906:�� 
block3_conv1_8908:	�-
block3_conv2_8923:�� 
block3_conv2_8925:	�-
block3_conv3_8940:�� 
block3_conv3_8942:	�-
block3_conv4_8957:�� 
block3_conv4_8959:	�-
block4_conv1_8980:�� 
block4_conv1_8982:	�-
block4_conv2_8997:�� 
block4_conv2_8999:	�-
block4_conv3_9014:�� 
block4_conv3_9016:	�-
block4_conv4_9031:�� 
block4_conv4_9033:	�-
block5_conv1_9054:�� 
block5_conv1_9056:	�-
block5_conv2_9071:�� 
block5_conv2_9073:	�-
block5_conv3_9088:�� 
block5_conv3_9090:	�-
block5_conv4_9105:�� 
block5_conv4_9107:	�
identity��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�$block2_conv1/StatefulPartitionedCall�$block2_conv2/StatefulPartitionedCall�$block3_conv1/StatefulPartitionedCall�$block3_conv2/StatefulPartitionedCall�$block3_conv3/StatefulPartitionedCall�$block3_conv4/StatefulPartitionedCall�$block4_conv1/StatefulPartitionedCall�$block4_conv2/StatefulPartitionedCall�$block4_conv3/StatefulPartitionedCall�$block4_conv4/StatefulPartitionedCall�$block5_conv1/StatefulPartitionedCall�$block5_conv2/StatefulPartitionedCall�$block5_conv3/StatefulPartitionedCall�$block5_conv4/StatefulPartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_8826block1_conv1_8828*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_88252&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_8843block1_conv2_8845*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_88422&
$block1_conv2/StatefulPartitionedCall�
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_88522
block1_pool/PartitionedCall�
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_8866block2_conv1_8868*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_88652&
$block2_conv1/StatefulPartitionedCall�
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_8883block2_conv2_8885*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_88822&
$block2_conv2/StatefulPartitionedCall�
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_88922
block2_pool/PartitionedCall�
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_8906block3_conv1_8908*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_89052&
$block3_conv1/StatefulPartitionedCall�
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_8923block3_conv2_8925*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_89222&
$block3_conv2/StatefulPartitionedCall�
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_8940block3_conv3_8942*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_89392&
$block3_conv3/StatefulPartitionedCall�
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_8957block3_conv4_8959*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_89562&
$block3_conv4/StatefulPartitionedCall�
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_89662
block3_pool/PartitionedCall�
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_8980block4_conv1_8982*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_89792&
$block4_conv1/StatefulPartitionedCall�
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_8997block4_conv2_8999*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_89962&
$block4_conv2/StatefulPartitionedCall�
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_9014block4_conv3_9016*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_90132&
$block4_conv3/StatefulPartitionedCall�
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_9031block4_conv4_9033*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_90302&
$block4_conv4/StatefulPartitionedCall�
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_90402
block4_pool/PartitionedCall�
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_9054block5_conv1_9056*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_90532&
$block5_conv1/StatefulPartitionedCall�
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_9071block5_conv2_9073*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_90702&
$block5_conv2/StatefulPartitionedCall�
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_9088block5_conv3_9090*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_90872&
$block5_conv3/StatefulPartitionedCall�
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_9105block5_conv4_9107*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_91042&
$block5_conv4/StatefulPartitionedCall�
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_91142
block5_pool/PartitionedCall�
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identity�
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
,__inference_block4_conv2_layer_call_fn_12183

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_89962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_block3_conv2_layer_call_fn_12083

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_89222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
F__inference_block4_conv4_layer_call_and_return_conditional_losses_9030

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block3_conv4_layer_call_and_return_conditional_losses_8956

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
B__inference_dense_3_layer_call_and_return_conditional_losses_11914

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
G
+__inference_block4_pool_layer_call_fn_12243

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_90402
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dense_3_layer_call_fn_11923

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_101292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�-
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_10445

inputs%
vgg19_10358:@
vgg19_10360:@%
vgg19_10362:@@
vgg19_10364:@&
vgg19_10366:@�
vgg19_10368:	�'
vgg19_10370:��
vgg19_10372:	�'
vgg19_10374:��
vgg19_10376:	�'
vgg19_10378:��
vgg19_10380:	�'
vgg19_10382:��
vgg19_10384:	�'
vgg19_10386:��
vgg19_10388:	�'
vgg19_10390:��
vgg19_10392:	�'
vgg19_10394:��
vgg19_10396:	�'
vgg19_10398:��
vgg19_10400:	�'
vgg19_10402:��
vgg19_10404:	�'
vgg19_10406:��
vgg19_10408:	�'
vgg19_10410:��
vgg19_10412:	�'
vgg19_10414:��
vgg19_10416:	�'
vgg19_10418:��
vgg19_10420:	�+
batch_normalization_1_10425:
��+
batch_normalization_1_10427:
��+
batch_normalization_1_10429:
��+
batch_normalization_1_10431:
��!
dense_2_10434:
��
dense_2_10436:
dense_3_10439:
dense_3_10441:
identity��-batch_normalization_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�vgg19/StatefulPartitionedCall�
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputsvgg19_10358vgg19_10360vgg19_10362vgg19_10364vgg19_10366vgg19_10368vgg19_10370vgg19_10372vgg19_10374vgg19_10376vgg19_10378vgg19_10380vgg19_10382vgg19_10384vgg19_10386vgg19_10388vgg19_10390vgg19_10392vgg19_10394vgg19_10396vgg19_10398vgg19_10400vgg19_10402vgg19_10404vgg19_10406vgg19_10408vgg19_10410vgg19_10412vgg19_10414vgg19_10416vgg19_10418vgg19_10420*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vgg19_layer_call_and_return_conditional_losses_95292
vgg19/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_100832
flatten_1/PartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_102592#
!dropout_1/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0batch_normalization_1_10425batch_normalization_1_10427batch_normalization_1_10429batch_normalization_1_10431*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_99272/
-batch_normalization_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_10434dense_2_10436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_101122!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_10439dense_3_10441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_101292!
dense_3/StatefulPartitionedCall�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
,__inference_block4_conv3_layer_call_fn_12203

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_90132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_block2_conv1_layer_call_fn_12003

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_88652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������pp�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������pp@
 
_user_specified_nameinputs
�
�
G__inference_block1_conv1_layer_call_and_return_conditional_losses_11934

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
G__inference_block3_conv2_layer_call_and_return_conditional_losses_12074

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_11973

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������pp@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
G__inference_block5_conv4_layer_call_and_return_conditional_losses_12314

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block4_conv1_layer_call_and_return_conditional_losses_8979

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_block5_pool_layer_call_fn_12338

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_87942
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_block3_conv1_layer_call_fn_12063

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_89052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
��
�
@__inference_vgg19_layer_call_and_return_conditional_losses_11627

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@�;
,block2_conv1_biasadd_readvariableop_resource:	�G
+block2_conv2_conv2d_readvariableop_resource:��;
,block2_conv2_biasadd_readvariableop_resource:	�G
+block3_conv1_conv2d_readvariableop_resource:��;
,block3_conv1_biasadd_readvariableop_resource:	�G
+block3_conv2_conv2d_readvariableop_resource:��;
,block3_conv2_biasadd_readvariableop_resource:	�G
+block3_conv3_conv2d_readvariableop_resource:��;
,block3_conv3_biasadd_readvariableop_resource:	�G
+block3_conv4_conv2d_readvariableop_resource:��;
,block3_conv4_biasadd_readvariableop_resource:	�G
+block4_conv1_conv2d_readvariableop_resource:��;
,block4_conv1_biasadd_readvariableop_resource:	�G
+block4_conv2_conv2d_readvariableop_resource:��;
,block4_conv2_biasadd_readvariableop_resource:	�G
+block4_conv3_conv2d_readvariableop_resource:��;
,block4_conv3_biasadd_readvariableop_resource:	�G
+block4_conv4_conv2d_readvariableop_resource:��;
,block4_conv4_biasadd_readvariableop_resource:	�G
+block5_conv1_conv2d_readvariableop_resource:��;
,block5_conv1_biasadd_readvariableop_resource:	�G
+block5_conv2_conv2d_readvariableop_resource:��;
,block5_conv2_biasadd_readvariableop_resource:	�G
+block5_conv3_conv2d_readvariableop_resource:��;
,block5_conv3_biasadd_readvariableop_resource:	�G
+block5_conv4_conv2d_readvariableop_resource:��;
,block5_conv4_biasadd_readvariableop_resource:	�
identity��#block1_conv1/BiasAdd/ReadVariableOp�"block1_conv1/Conv2D/ReadVariableOp�#block1_conv2/BiasAdd/ReadVariableOp�"block1_conv2/Conv2D/ReadVariableOp�#block2_conv1/BiasAdd/ReadVariableOp�"block2_conv1/Conv2D/ReadVariableOp�#block2_conv2/BiasAdd/ReadVariableOp�"block2_conv2/Conv2D/ReadVariableOp�#block3_conv1/BiasAdd/ReadVariableOp�"block3_conv1/Conv2D/ReadVariableOp�#block3_conv2/BiasAdd/ReadVariableOp�"block3_conv2/Conv2D/ReadVariableOp�#block3_conv3/BiasAdd/ReadVariableOp�"block3_conv3/Conv2D/ReadVariableOp�#block3_conv4/BiasAdd/ReadVariableOp�"block3_conv4/Conv2D/ReadVariableOp�#block4_conv1/BiasAdd/ReadVariableOp�"block4_conv1/Conv2D/ReadVariableOp�#block4_conv2/BiasAdd/ReadVariableOp�"block4_conv2/Conv2D/ReadVariableOp�#block4_conv3/BiasAdd/ReadVariableOp�"block4_conv3/Conv2D/ReadVariableOp�#block4_conv4/BiasAdd/ReadVariableOp�"block4_conv4/Conv2D/ReadVariableOp�#block5_conv1/BiasAdd/ReadVariableOp�"block5_conv1/Conv2D/ReadVariableOp�#block5_conv2/BiasAdd/ReadVariableOp�"block5_conv2/Conv2D/ReadVariableOp�#block5_conv3/BiasAdd/ReadVariableOp�"block5_conv3/Conv2D/ReadVariableOp�#block5_conv4/BiasAdd/ReadVariableOp�"block5_conv4/Conv2D/ReadVariableOp�
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp�
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
block1_conv1/Conv2D�
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp�
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2
block1_conv1/BiasAdd�
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
block1_conv1/Relu�
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp�
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
block1_conv2/Conv2D�
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp�
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2
block1_conv2/BiasAdd�
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
block1_conv2/Relu�
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool�
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp�
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
block2_conv1/Conv2D�
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp�
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2
block2_conv1/BiasAdd�
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
block2_conv1/Relu�
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp�
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
block2_conv2/Conv2D�
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp�
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2
block2_conv2/BiasAdd�
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
block2_conv2/Relu�
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:���������88�*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool�
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp�
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
block3_conv1/Conv2D�
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp�
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
block3_conv1/BiasAdd�
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
block3_conv1/Relu�
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp�
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
block3_conv2/Conv2D�
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp�
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
block3_conv2/BiasAdd�
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
block3_conv2/Relu�
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp�
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
block3_conv3/Conv2D�
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp�
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
block3_conv3/BiasAdd�
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
block3_conv3/Relu�
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block3_conv4/Conv2D/ReadVariableOp�
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
block3_conv4/Conv2D�
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block3_conv4/BiasAdd/ReadVariableOp�
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2
block3_conv4/BiasAdd�
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
block3_conv4/Relu�
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool�
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp�
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block4_conv1/Conv2D�
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp�
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block4_conv1/BiasAdd�
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block4_conv1/Relu�
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp�
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block4_conv2/Conv2D�
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp�
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block4_conv2/BiasAdd�
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block4_conv2/Relu�
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp�
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block4_conv3/Conv2D�
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp�
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block4_conv3/BiasAdd�
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block4_conv3/Relu�
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block4_conv4/Conv2D/ReadVariableOp�
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block4_conv4/Conv2D�
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block4_conv4/BiasAdd/ReadVariableOp�
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block4_conv4/BiasAdd�
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block4_conv4/Relu�
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool�
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp�
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block5_conv1/Conv2D�
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp�
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block5_conv1/BiasAdd�
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block5_conv1/Relu�
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block5_conv2/Conv2D/ReadVariableOp�
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block5_conv2/Conv2D�
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp�
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block5_conv2/BiasAdd�
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block5_conv2/Relu�
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block5_conv3/Conv2D/ReadVariableOp�
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block5_conv3/Conv2D�
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp�
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block5_conv3/BiasAdd�
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block5_conv3/Relu�
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02$
"block5_conv4/Conv2D/ReadVariableOp�
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
block5_conv4/Conv2D�
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#block5_conv4/BiasAdd/ReadVariableOp�
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
block5_conv4/BiasAdd�
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
block5_conv4/Relu�
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool�
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identity�	
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_block4_conv2_layer_call_and_return_conditional_losses_8996

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_block4_conv1_layer_call_and_return_conditional_losses_12154

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
�&
!__inference__traced_restore_12742
file_prefix<
,assignvariableop_batch_normalization_1_gamma:
��=
-assignvariableop_1_batch_normalization_1_beta:
��D
4assignvariableop_2_batch_normalization_1_moving_mean:
��H
8assignvariableop_3_batch_normalization_1_moving_variance:
��5
!assignvariableop_4_dense_2_kernel:
��-
assignvariableop_5_dense_2_bias:3
!assignvariableop_6_dense_3_kernel:-
assignvariableop_7_dense_3_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: A
'assignvariableop_13_block1_conv1_kernel:@3
%assignvariableop_14_block1_conv1_bias:@A
'assignvariableop_15_block1_conv2_kernel:@@3
%assignvariableop_16_block1_conv2_bias:@B
'assignvariableop_17_block2_conv1_kernel:@�4
%assignvariableop_18_block2_conv1_bias:	�C
'assignvariableop_19_block2_conv2_kernel:��4
%assignvariableop_20_block2_conv2_bias:	�C
'assignvariableop_21_block3_conv1_kernel:��4
%assignvariableop_22_block3_conv1_bias:	�C
'assignvariableop_23_block3_conv2_kernel:��4
%assignvariableop_24_block3_conv2_bias:	�C
'assignvariableop_25_block3_conv3_kernel:��4
%assignvariableop_26_block3_conv3_bias:	�C
'assignvariableop_27_block3_conv4_kernel:��4
%assignvariableop_28_block3_conv4_bias:	�C
'assignvariableop_29_block4_conv1_kernel:��4
%assignvariableop_30_block4_conv1_bias:	�C
'assignvariableop_31_block4_conv2_kernel:��4
%assignvariableop_32_block4_conv2_bias:	�C
'assignvariableop_33_block4_conv3_kernel:��4
%assignvariableop_34_block4_conv3_bias:	�C
'assignvariableop_35_block4_conv4_kernel:��4
%assignvariableop_36_block4_conv4_bias:	�C
'assignvariableop_37_block5_conv1_kernel:��4
%assignvariableop_38_block5_conv1_bias:	�C
'assignvariableop_39_block5_conv2_kernel:��4
%assignvariableop_40_block5_conv2_bias:	�C
'assignvariableop_41_block5_conv3_kernel:��4
%assignvariableop_42_block5_conv3_bias:	�C
'assignvariableop_43_block5_conv4_kernel:��4
%assignvariableop_44_block5_conv4_bias:	�#
assignvariableop_45_total: #
assignvariableop_46_count: %
assignvariableop_47_total_1: %
assignvariableop_48_count_1: F
6assignvariableop_49_adam_batch_normalization_1_gamma_m:
��E
5assignvariableop_50_adam_batch_normalization_1_beta_m:
��=
)assignvariableop_51_adam_dense_2_kernel_m:
��5
'assignvariableop_52_adam_dense_2_bias_m:;
)assignvariableop_53_adam_dense_3_kernel_m:5
'assignvariableop_54_adam_dense_3_bias_m:F
6assignvariableop_55_adam_batch_normalization_1_gamma_v:
��E
5assignvariableop_56_adam_batch_normalization_1_beta_v:
��=
)assignvariableop_57_adam_dense_2_kernel_v:
��5
'assignvariableop_58_adam_dense_2_bias_v:;
)assignvariableop_59_adam_dense_3_kernel_v:5
'assignvariableop_60_adam_dense_3_bias_v:
identity_62��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*�
value�B�>B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*�
value�B�>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*L
dtypesB
@2>	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp,assignvariableop_batch_normalization_1_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp-assignvariableop_1_batch_normalization_1_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp4assignvariableop_2_batch_normalization_1_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp8assignvariableop_3_batch_normalization_1_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp'assignvariableop_13_block1_conv1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp%assignvariableop_14_block1_conv1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_block1_conv2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp%assignvariableop_16_block1_conv2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp'assignvariableop_17_block2_conv1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp%assignvariableop_18_block2_conv1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_block2_conv2_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_block2_conv2_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp'assignvariableop_21_block3_conv1_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp%assignvariableop_22_block3_conv1_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_block3_conv2_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp%assignvariableop_24_block3_conv2_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_block3_conv3_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp%assignvariableop_26_block3_conv3_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_block3_conv4_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_block3_conv4_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_block4_conv1_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp%assignvariableop_30_block4_conv1_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp'assignvariableop_31_block4_conv2_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp%assignvariableop_32_block4_conv2_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp'assignvariableop_33_block4_conv3_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp%assignvariableop_34_block4_conv3_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_block4_conv4_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp%assignvariableop_36_block4_conv4_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp'assignvariableop_37_block5_conv1_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp%assignvariableop_38_block5_conv1_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_block5_conv2_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp%assignvariableop_40_block5_conv2_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp'assignvariableop_41_block5_conv3_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp%assignvariableop_42_block5_conv3_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp'assignvariableop_43_block5_conv4_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp%assignvariableop_44_block5_conv4_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOpassignvariableop_45_totalIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOpassignvariableop_46_countIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_1_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_1_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_2_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_2_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_3_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_3_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_1_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_batch_normalization_1_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp)assignvariableop_57_adam_dense_2_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_dense_2_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_3_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_3_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_609
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_61Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_61f
Identity_62IdentityIdentity_61:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_62�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_62Identity_62:output:0*�
_input_shapes~
|: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
F__inference_block4_conv3_layer_call_and_return_conditional_losses_9013

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_block4_conv4_layer_call_and_return_conditional_losses_12214

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_block1_conv1_layer_call_fn_11943

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_88252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�l
�
?__inference_vgg19_layer_call_and_return_conditional_losses_9529

inputs+
block1_conv1_9443:@
block1_conv1_9445:@+
block1_conv2_9448:@@
block1_conv2_9450:@,
block2_conv1_9454:@� 
block2_conv1_9456:	�-
block2_conv2_9459:�� 
block2_conv2_9461:	�-
block3_conv1_9465:�� 
block3_conv1_9467:	�-
block3_conv2_9470:�� 
block3_conv2_9472:	�-
block3_conv3_9475:�� 
block3_conv3_9477:	�-
block3_conv4_9480:�� 
block3_conv4_9482:	�-
block4_conv1_9486:�� 
block4_conv1_9488:	�-
block4_conv2_9491:�� 
block4_conv2_9493:	�-
block4_conv3_9496:�� 
block4_conv3_9498:	�-
block4_conv4_9501:�� 
block4_conv4_9503:	�-
block5_conv1_9507:�� 
block5_conv1_9509:	�-
block5_conv2_9512:�� 
block5_conv2_9514:	�-
block5_conv3_9517:�� 
block5_conv3_9519:	�-
block5_conv4_9522:�� 
block5_conv4_9524:	�
identity��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�$block2_conv1/StatefulPartitionedCall�$block2_conv2/StatefulPartitionedCall�$block3_conv1/StatefulPartitionedCall�$block3_conv2/StatefulPartitionedCall�$block3_conv3/StatefulPartitionedCall�$block3_conv4/StatefulPartitionedCall�$block4_conv1/StatefulPartitionedCall�$block4_conv2/StatefulPartitionedCall�$block4_conv3/StatefulPartitionedCall�$block4_conv4/StatefulPartitionedCall�$block5_conv1/StatefulPartitionedCall�$block5_conv2/StatefulPartitionedCall�$block5_conv3/StatefulPartitionedCall�$block5_conv4/StatefulPartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_9443block1_conv1_9445*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_88252&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_9448block1_conv2_9450*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_88422&
$block1_conv2/StatefulPartitionedCall�
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_88522
block1_pool/PartitionedCall�
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_9454block2_conv1_9456*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_88652&
$block2_conv1/StatefulPartitionedCall�
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_9459block2_conv2_9461*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_88822&
$block2_conv2/StatefulPartitionedCall�
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_88922
block2_pool/PartitionedCall�
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_9465block3_conv1_9467*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_89052&
$block3_conv1/StatefulPartitionedCall�
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_9470block3_conv2_9472*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_89222&
$block3_conv2/StatefulPartitionedCall�
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_9475block3_conv3_9477*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_89392&
$block3_conv3/StatefulPartitionedCall�
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_9480block3_conv4_9482*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_89562&
$block3_conv4/StatefulPartitionedCall�
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_89662
block3_pool/PartitionedCall�
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_9486block4_conv1_9488*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_89792&
$block4_conv1/StatefulPartitionedCall�
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_9491block4_conv2_9493*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_89962&
$block4_conv2/StatefulPartitionedCall�
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_9496block4_conv3_9498*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_90132&
$block4_conv3/StatefulPartitionedCall�
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_9501block4_conv4_9503*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_90302&
$block4_conv4/StatefulPartitionedCall�
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_90402
block4_pool/PartitionedCall�
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_9507block5_conv1_9509*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_90532&
$block5_conv1/StatefulPartitionedCall�
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_9512block5_conv2_9514*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_90702&
$block5_conv2/StatefulPartitionedCall�
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_9517block5_conv3_9519*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_90872&
$block5_conv3/StatefulPartitionedCall�
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_9522block5_conv4_9524*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_91042&
$block5_conv4/StatefulPartitionedCall�
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_91142
block5_pool/PartitionedCall�
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identity�
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
G__inference_block3_conv1_layer_call_and_return_conditional_losses_12054

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_8966

inputs
identity�
MaxPoolMaxPoolinputs*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������88�:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
F__inference_block5_conv1_layer_call_and_return_conditional_losses_9053

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9867

inputs1
!batchnorm_readvariableop_resource:
��5
%batchnorm_mul_readvariableop_resource:
��3
#batchnorm_readvariableop_1_resource:
��3
#batchnorm_readvariableop_2_resource:
��
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes

:��*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes

:��2
batchnorm/adde
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes

:��2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes

:��*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:��2
batchnorm/mulx
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*)
_output_shapes
:�����������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes

:��*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes

:��2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes

:��*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes

:��2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*)
_output_shapes
:�����������2
batchnorm/add_1p
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*)
_output_shapes
:�����������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_block1_conv2_layer_call_and_return_conditional_losses_8842

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
F__inference_block1_conv1_layer_call_and_return_conditional_losses_8825

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_10083

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_vgg19_layer_call_fn_9665
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vgg19_layer_call_and_return_conditional_losses_95292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_8706

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_11771

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block3_conv3_layer_call_and_return_conditional_losses_8939

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_8794

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_8892

inputs
identity�
MaxPoolMaxPoolinputs*0
_output_shapes
:���������88�*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:���������88�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������pp�:X T
0
_output_shapes
:���������pp�
 
_user_specified_nameinputs
�,
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_10703
vgg19_input%
vgg19_10616:@
vgg19_10618:@%
vgg19_10620:@@
vgg19_10622:@&
vgg19_10624:@�
vgg19_10626:	�'
vgg19_10628:��
vgg19_10630:	�'
vgg19_10632:��
vgg19_10634:	�'
vgg19_10636:��
vgg19_10638:	�'
vgg19_10640:��
vgg19_10642:	�'
vgg19_10644:��
vgg19_10646:	�'
vgg19_10648:��
vgg19_10650:	�'
vgg19_10652:��
vgg19_10654:	�'
vgg19_10656:��
vgg19_10658:	�'
vgg19_10660:��
vgg19_10662:	�'
vgg19_10664:��
vgg19_10666:	�'
vgg19_10668:��
vgg19_10670:	�'
vgg19_10672:��
vgg19_10674:	�'
vgg19_10676:��
vgg19_10678:	�+
batch_normalization_1_10683:
��+
batch_normalization_1_10685:
��+
batch_normalization_1_10687:
��+
batch_normalization_1_10689:
��!
dense_2_10692:
��
dense_2_10694:
dense_3_10697:
dense_3_10699:
identity��-batch_normalization_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�vgg19/StatefulPartitionedCall�
vgg19/StatefulPartitionedCallStatefulPartitionedCallvgg19_inputvgg19_10616vgg19_10618vgg19_10620vgg19_10622vgg19_10624vgg19_10626vgg19_10628vgg19_10630vgg19_10632vgg19_10634vgg19_10636vgg19_10638vgg19_10640vgg19_10642vgg19_10644vgg19_10646vgg19_10648vgg19_10650vgg19_10652vgg19_10654vgg19_10656vgg19_10658vgg19_10660vgg19_10662vgg19_10664vgg19_10666vgg19_10668vgg19_10670vgg19_10672vgg19_10674vgg19_10676vgg19_10678*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vgg19_layer_call_and_return_conditional_losses_91172
vgg19/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_100832
flatten_1/PartitionedCall�
dropout_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_100902
dropout_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0batch_normalization_1_10683batch_normalization_1_10685batch_normalization_1_10687batch_normalization_1_10689*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_98672/
-batch_normalization_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_10692dense_2_10694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_101122!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_10697dense_3_10699*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_101292!
dense_3/StatefulPartitionedCall�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg19_input
�
�
,__inference_block5_conv2_layer_call_fn_12283

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_90702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_9114

inputs
identity�
MaxPoolMaxPoolinputs*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_vgg19_layer_call_fn_11696

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vgg19_layer_call_and_return_conditional_losses_91172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�

,__inference_sequential_1_layer_call_fn_11385

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�

unknown_31:
��

unknown_32:
��

unknown_33:
��

unknown_34:
��

unknown_35:
��

unknown_36:

unknown_37:

unknown_38:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*H
_read_only_resource_inputs*
(&	
 #$%&'(*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_104452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11823

inputs1
!batchnorm_readvariableop_resource:
��5
%batchnorm_mul_readvariableop_resource:
��3
#batchnorm_readvariableop_1_resource:
��3
#batchnorm_readvariableop_2_resource:
��
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes

:��*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes

:��2
batchnorm/adde
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes

:��2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes

:��*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:��2
batchnorm/mulx
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*)
_output_shapes
:�����������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes

:��*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes

:��2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes

:��*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes

:��2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*)
_output_shapes
:�����������2
batchnorm/add_1p
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*)
_output_shapes
:�����������2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_12033

inputs
identity�
MaxPoolMaxPoolinputs*0
_output_shapes
:���������88�*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:���������88�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������pp�:X T
0
_output_shapes
:���������pp�
 
_user_specified_nameinputs
�
b
)__inference_dropout_1_layer_call_fn_11803

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_102592
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_8772

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�s
�
__inference__traced_save_12549
file_prefix:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block3_conv4_kernel_read_readvariableop0
,savev2_block3_conv4_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block4_conv4_kernel_read_readvariableop0
,savev2_block4_conv4_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop2
.savev2_block5_conv4_kernel_read_readvariableop0
,savev2_block5_conv4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*�
value�B�>B5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*�
value�B�>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block3_conv4_kernel_read_readvariableop,savev2_block3_conv4_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block4_conv4_kernel_read_readvariableop,savev2_block4_conv4_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop.savev2_block5_conv4_kernel_read_readvariableop,savev2_block5_conv4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *L
dtypesB
@2>	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :��:��:��:��:
��:::: : : : : :@:@:@@:@:@�:�:��:�:��:�:��:�:��:�:��:�:��:�:��:�:��:�:��:�:��:�:��:�:��:�:��:�: : : : :��:��:
��::::��:��:
��:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:"

_output_shapes

:��:"

_output_shapes

:��:"

_output_shapes

:��:"

_output_shapes

:��:&"
 
_output_shapes
:
��: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:. *
(
_output_shapes
:��:!!

_output_shapes	
:�:."*
(
_output_shapes
:��:!#

_output_shapes	
:�:.$*
(
_output_shapes
:��:!%

_output_shapes	
:�:.&*
(
_output_shapes
:��:!'

_output_shapes	
:�:.(*
(
_output_shapes
:��:!)

_output_shapes	
:�:.**
(
_output_shapes
:��:!+

_output_shapes	
:�:.,*
(
_output_shapes
:��:!-

_output_shapes	
:�:.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :"2

_output_shapes

:��:"3

_output_shapes

:��:&4"
 
_output_shapes
:
��: 5

_output_shapes
::$6 

_output_shapes

:: 7

_output_shapes
::"8

_output_shapes

:��:"9

_output_shapes

:��:&:"
 
_output_shapes
:
��: ;

_output_shapes
::$< 

_output_shapes

:: =

_output_shapes
::>

_output_shapes
: 
�
G
+__inference_block2_pool_layer_call_fn_12038

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_87282
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_8750

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�+
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11857

inputs7
'assignmovingavg_readvariableop_resource:
��9
)assignmovingavg_1_readvariableop_resource:
��5
%batchnorm_mul_readvariableop_resource:
��1
!batchnorm_readvariableop_resource:
��
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0* 
_output_shapes
:
��*
	keep_dims(2
moments/mean~
moments/StopGradientStopGradientmoments/mean:output:0*
T0* 
_output_shapes
:
��2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*)
_output_shapes
:�����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0* 
_output_shapes
:
��*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes

:��*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes

:��*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes

:��*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes

:��2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes

:��2
AssignMovingAvg/mul�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes

:��*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes

:��2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes

:��2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes

:��2
batchnorm/adde
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes

:��2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes

:��*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:��2
batchnorm/mulx
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*)
_output_shapes
:�����������2
batchnorm/mul_1}
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes

:��2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes

:��*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes

:��2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*)
_output_shapes
:�����������2
batchnorm/add_1p
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*)
_output_shapes
:�����������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_block3_conv2_layer_call_and_return_conditional_losses_8922

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_12233

inputs
identity�
MaxPoolMaxPoolinputs*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block5_conv3_layer_call_and_return_conditional_losses_9087

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_8852

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������pp@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_9040

inputs
identity�
MaxPoolMaxPoolinputs*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_block4_conv1_layer_call_fn_12163

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_89792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_1_layer_call_fn_11870

inputs
unknown:
��
	unknown_0:
��
	unknown_1:
��
	unknown_2:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_98672
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_11793

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constu
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:�����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:�����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:�����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:�����������2
dropout/Cast|
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:�����������2
dropout/Mul_1g
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
G
+__inference_block3_pool_layer_call_fn_12138

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_87502
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_block5_conv4_layer_call_fn_12323

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_91042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_flatten_1_layer_call_fn_11776

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_100832
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�

,__inference_sequential_1_layer_call_fn_11300

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�

unknown_31:
��

unknown_32:
��

unknown_33:
��

unknown_34:
��

unknown_35:
��

unknown_36:

unknown_37:

unknown_38:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_101362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_block5_conv2_layer_call_and_return_conditional_losses_9070

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�l
�
?__inference_vgg19_layer_call_and_return_conditional_losses_9754
input_2+
block1_conv1_9668:@
block1_conv1_9670:@+
block1_conv2_9673:@@
block1_conv2_9675:@,
block2_conv1_9679:@� 
block2_conv1_9681:	�-
block2_conv2_9684:�� 
block2_conv2_9686:	�-
block3_conv1_9690:�� 
block3_conv1_9692:	�-
block3_conv2_9695:�� 
block3_conv2_9697:	�-
block3_conv3_9700:�� 
block3_conv3_9702:	�-
block3_conv4_9705:�� 
block3_conv4_9707:	�-
block4_conv1_9711:�� 
block4_conv1_9713:	�-
block4_conv2_9716:�� 
block4_conv2_9718:	�-
block4_conv3_9721:�� 
block4_conv3_9723:	�-
block4_conv4_9726:�� 
block4_conv4_9728:	�-
block5_conv1_9732:�� 
block5_conv1_9734:	�-
block5_conv2_9737:�� 
block5_conv2_9739:	�-
block5_conv3_9742:�� 
block5_conv3_9744:	�-
block5_conv4_9747:�� 
block5_conv4_9749:	�
identity��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�$block2_conv1/StatefulPartitionedCall�$block2_conv2/StatefulPartitionedCall�$block3_conv1/StatefulPartitionedCall�$block3_conv2/StatefulPartitionedCall�$block3_conv3/StatefulPartitionedCall�$block3_conv4/StatefulPartitionedCall�$block4_conv1/StatefulPartitionedCall�$block4_conv2/StatefulPartitionedCall�$block4_conv3/StatefulPartitionedCall�$block4_conv4/StatefulPartitionedCall�$block5_conv1/StatefulPartitionedCall�$block5_conv2/StatefulPartitionedCall�$block5_conv3/StatefulPartitionedCall�$block5_conv4/StatefulPartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_9668block1_conv1_9670*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_88252&
$block1_conv1/StatefulPartitionedCall�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_9673block1_conv2_9675*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_88422&
$block1_conv2/StatefulPartitionedCall�
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_88522
block1_pool/PartitionedCall�
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_9679block2_conv1_9681*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_88652&
$block2_conv1/StatefulPartitionedCall�
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_9684block2_conv2_9686*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_88822&
$block2_conv2/StatefulPartitionedCall�
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_88922
block2_pool/PartitionedCall�
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_9690block3_conv1_9692*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_89052&
$block3_conv1/StatefulPartitionedCall�
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_9695block3_conv2_9697*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_89222&
$block3_conv2/StatefulPartitionedCall�
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_9700block3_conv3_9702*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_89392&
$block3_conv3/StatefulPartitionedCall�
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_9705block3_conv4_9707*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_89562&
$block3_conv4/StatefulPartitionedCall�
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_89662
block3_pool/PartitionedCall�
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_9711block4_conv1_9713*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_89792&
$block4_conv1/StatefulPartitionedCall�
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_9716block4_conv2_9718*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_89962&
$block4_conv2/StatefulPartitionedCall�
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_9721block4_conv3_9723*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_90132&
$block4_conv3/StatefulPartitionedCall�
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_9726block4_conv4_9728*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block4_conv4_layer_call_and_return_conditional_losses_90302&
$block4_conv4/StatefulPartitionedCall�
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_90402
block4_pool/PartitionedCall�
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_9732block5_conv1_9734*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_90532&
$block5_conv1/StatefulPartitionedCall�
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_9737block5_conv2_9739*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_90702&
$block5_conv2/StatefulPartitionedCall�
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_9742block5_conv3_9744*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_90872&
$block5_conv3/StatefulPartitionedCall�
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_9747block5_conv4_9749*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block5_conv4_layer_call_and_return_conditional_losses_91042&
$block5_conv4/StatefulPartitionedCall�
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_91142
block5_pool/PartitionedCall�
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identity�
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�,
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_10136

inputs%
vgg19_10012:@
vgg19_10014:@%
vgg19_10016:@@
vgg19_10018:@&
vgg19_10020:@�
vgg19_10022:	�'
vgg19_10024:��
vgg19_10026:	�'
vgg19_10028:��
vgg19_10030:	�'
vgg19_10032:��
vgg19_10034:	�'
vgg19_10036:��
vgg19_10038:	�'
vgg19_10040:��
vgg19_10042:	�'
vgg19_10044:��
vgg19_10046:	�'
vgg19_10048:��
vgg19_10050:	�'
vgg19_10052:��
vgg19_10054:	�'
vgg19_10056:��
vgg19_10058:	�'
vgg19_10060:��
vgg19_10062:	�'
vgg19_10064:��
vgg19_10066:	�'
vgg19_10068:��
vgg19_10070:	�'
vgg19_10072:��
vgg19_10074:	�+
batch_normalization_1_10092:
��+
batch_normalization_1_10094:
��+
batch_normalization_1_10096:
��+
batch_normalization_1_10098:
��!
dense_2_10113:
��
dense_2_10115:
dense_3_10130:
dense_3_10132:
identity��-batch_normalization_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�vgg19/StatefulPartitionedCall�
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputsvgg19_10012vgg19_10014vgg19_10016vgg19_10018vgg19_10020vgg19_10022vgg19_10024vgg19_10026vgg19_10028vgg19_10030vgg19_10032vgg19_10034vgg19_10036vgg19_10038vgg19_10040vgg19_10042vgg19_10044vgg19_10046vgg19_10048vgg19_10050vgg19_10052vgg19_10054vgg19_10056vgg19_10058vgg19_10060vgg19_10062vgg19_10064vgg19_10066vgg19_10068vgg19_10070vgg19_10072vgg19_10074*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vgg19_layer_call_and_return_conditional_losses_91172
vgg19/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_100832
flatten_1/PartitionedCall�
dropout_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_100902
dropout_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0batch_normalization_1_10092batch_normalization_1_10094batch_normalization_1_10096batch_normalization_1_10098*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_98672/
-batch_normalization_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_10113dense_2_10115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_101122!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_10130dense_3_10132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_101292!
dense_3/StatefulPartitionedCall�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
G__inference_block2_conv2_layer_call_and_return_conditional_losses_12014

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������pp�2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������pp�2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������pp�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������pp�
 
_user_specified_nameinputs
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_11781

inputs

identity_1\
IdentityIdentityinputs*
T0*)
_output_shapes
:�����������2

Identityk

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:�����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�-
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_10793
vgg19_input%
vgg19_10706:@
vgg19_10708:@%
vgg19_10710:@@
vgg19_10712:@&
vgg19_10714:@�
vgg19_10716:	�'
vgg19_10718:��
vgg19_10720:	�'
vgg19_10722:��
vgg19_10724:	�'
vgg19_10726:��
vgg19_10728:	�'
vgg19_10730:��
vgg19_10732:	�'
vgg19_10734:��
vgg19_10736:	�'
vgg19_10738:��
vgg19_10740:	�'
vgg19_10742:��
vgg19_10744:	�'
vgg19_10746:��
vgg19_10748:	�'
vgg19_10750:��
vgg19_10752:	�'
vgg19_10754:��
vgg19_10756:	�'
vgg19_10758:��
vgg19_10760:	�'
vgg19_10762:��
vgg19_10764:	�'
vgg19_10766:��
vgg19_10768:	�+
batch_normalization_1_10773:
��+
batch_normalization_1_10775:
��+
batch_normalization_1_10777:
��+
batch_normalization_1_10779:
��!
dense_2_10782:
��
dense_2_10784:
dense_3_10787:
dense_3_10789:
identity��-batch_normalization_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�vgg19/StatefulPartitionedCall�
vgg19/StatefulPartitionedCallStatefulPartitionedCallvgg19_inputvgg19_10706vgg19_10708vgg19_10710vgg19_10712vgg19_10714vgg19_10716vgg19_10718vgg19_10720vgg19_10722vgg19_10724vgg19_10726vgg19_10728vgg19_10730vgg19_10732vgg19_10734vgg19_10736vgg19_10738vgg19_10740vgg19_10742vgg19_10744vgg19_10746vgg19_10748vgg19_10750vgg19_10752vgg19_10754vgg19_10756vgg19_10758vgg19_10760vgg19_10762vgg19_10764vgg19_10766vgg19_10768*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_vgg19_layer_call_and_return_conditional_losses_95292
vgg19/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_100832
flatten_1/PartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_102592#
!dropout_1/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0batch_normalization_1_10773batch_normalization_1_10775batch_normalization_1_10777batch_normalization_1_10779*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_99272/
-batch_normalization_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_2_10782dense_2_10784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_101122!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_10787dense_3_10789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_101292!
dense_3/StatefulPartitionedCall�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg19_input
�
�
,__inference_block3_conv3_layer_call_fn_12103

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_89392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
G
+__inference_block1_pool_layer_call_fn_11978

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_87062
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_block3_conv4_layer_call_fn_12123

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block3_conv4_layer_call_and_return_conditional_losses_89562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������88�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
,__inference_block2_conv2_layer_call_fn_12023

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_88822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������pp�2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������pp�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������pp�
 
_user_specified_nameinputs
�
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_12128

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
G__inference_block4_conv3_layer_call_and_return_conditional_losses_12194

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dense_2_layer_call_fn_11903

inputs
unknown:
��
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_101122
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_1_layer_call_fn_11883

inputs
unknown:
��
	unknown_0:
��
	unknown_1:
��
	unknown_2:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_99272
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
G
+__inference_block4_pool_layer_call_fn_12238

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_87722
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�

,__inference_sequential_1_layer_call_fn_10219
vgg19_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�&

unknown_25:��

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�

unknown_31:
��

unknown_32:
��

unknown_33:
��

unknown_34:
��

unknown_35:
��

unknown_36:

unknown_37:

unknown_38:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvgg19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_101362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg19_input
�
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_8728

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
vgg19_input>
serving_default_vgg19_input:0�����������;
dense_30
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	optimizer
trainable_variables
		variables

regularization_losses
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"
_tf_keras_sequential
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
 layer_with_weights-14
 layer-19
!layer_with_weights-15
!layer-20
"layer-21
#trainable_variables
$	variables
%regularization_losses
&	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_network
�
'trainable_variables
(	variables
)regularization_losses
*	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
+trainable_variables
,	variables
-regularization_losses
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5	variables
6regularization_losses
7	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

>kernel
?bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_rate0m�1m�8m�9m�>m�?m�0v�1v�8v�9v�>v�?v�"
	optimizer
J
00
11
82
93
>4
?5"
trackable_list_wrapper
�
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19
]20
^21
_22
`23
a24
b25
c26
d27
e28
f29
g30
h31
032
133
234
335
836
937
>38
?39"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
inon_trainable_variables
jmetrics
klayer_regularization_losses
		variables

regularization_losses
llayer_metrics

mlayers
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
"
_tf_keras_input_layer
�

Ikernel
Jbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Kkernel
Lbias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Mkernel
Nbias
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Okernel
Pbias
~trainable_variables
	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Qkernel
Rbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Skernel
Tbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Ukernel
Vbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Wkernel
Xbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Ykernel
Zbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

[kernel
\bias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

]kernel
^bias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

_kernel
`bias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

akernel
bbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

ckernel
dbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

ekernel
fbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

gkernel
hbias
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
�trainable_variables
�	variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
 "
trackable_list_wrapper
�
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19
]20
^21
_22
`23
a24
b25
c26
d27
e28
f29
g30
h31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
#trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
$	variables
%regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
'trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
(	variables
)regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
+trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
,	variables
-regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)��2batch_normalization_1/gamma
*:(��2batch_normalization_1/beta
3:1�� (2!batch_normalization_1/moving_mean
7:5�� (2%batch_normalization_1/moving_variance
.
00
11"
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
�
4trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
5	variables
6regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 
��2dense_2/kernel
:2dense_2/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
:trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
;	variables
<regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
A	variables
Bregularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@�2block2_conv1/kernel
 :�2block2_conv1/bias
/:-��2block2_conv2/kernel
 :�2block2_conv2/bias
/:-��2block3_conv1/kernel
 :�2block3_conv1/bias
/:-��2block3_conv2/kernel
 :�2block3_conv2/bias
/:-��2block3_conv3/kernel
 :�2block3_conv3/bias
/:-��2block3_conv4/kernel
 :�2block3_conv4/bias
/:-��2block4_conv1/kernel
 :�2block4_conv1/bias
/:-��2block4_conv2/kernel
 :�2block4_conv2/bias
/:-��2block4_conv3/kernel
 :�2block4_conv3/bias
/:-��2block4_conv4/kernel
 :�2block4_conv4/bias
/:-��2block5_conv1/kernel
 :�2block5_conv1/bias
/:-��2block5_conv2/kernel
 :�2block5_conv2/bias
/:-��2block5_conv3/kernel
 :�2block5_conv3/bias
/:-��2block5_conv4/kernel
 :�2block5_conv4/bias
�
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19
]20
^21
_22
`23
a24
b25
c26
d27
e28
f29
g30
h31
232
333"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ntrainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
o	variables
pregularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
rtrainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
s	variables
tregularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
vtrainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
w	variables
xregularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ztrainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
{	variables
|regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
~trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
�metrics
 �layer_regularization_losses
�	variables
�regularization_losses
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
I0
J1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17
[18
\19
]20
^21
_22
`23
a24
b25
c26
d27
e28
f29
g30
h31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
0:.��2"Adam/batch_normalization_1/gamma/m
/:-��2!Adam/batch_normalization_1/beta/m
':%
��2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:#2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
0:.��2"Adam/batch_normalization_1/gamma/v
/:-��2!Adam/batch_normalization_1/beta/v
':%
��2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
%:#2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
�2�
G__inference_sequential_1_layer_call_and_return_conditional_losses_11040
G__inference_sequential_1_layer_call_and_return_conditional_losses_11215
G__inference_sequential_1_layer_call_and_return_conditional_losses_10703
G__inference_sequential_1_layer_call_and_return_conditional_losses_10793�
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
�2�
,__inference_sequential_1_layer_call_fn_10219
,__inference_sequential_1_layer_call_fn_11300
,__inference_sequential_1_layer_call_fn_11385
,__inference_sequential_1_layer_call_fn_10613�
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
�B�
__inference__wrapped_model_8697vgg19_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_vgg19_layer_call_and_return_conditional_losses_11506
@__inference_vgg19_layer_call_and_return_conditional_losses_11627
?__inference_vgg19_layer_call_and_return_conditional_losses_9754
?__inference_vgg19_layer_call_and_return_conditional_losses_9843�
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
�2�
$__inference_vgg19_layer_call_fn_9184
%__inference_vgg19_layer_call_fn_11696
%__inference_vgg19_layer_call_fn_11765
$__inference_vgg19_layer_call_fn_9665�
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
�2�
D__inference_flatten_1_layer_call_and_return_conditional_losses_11771�
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
)__inference_flatten_1_layer_call_fn_11776�
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
�2�
D__inference_dropout_1_layer_call_and_return_conditional_losses_11781
D__inference_dropout_1_layer_call_and_return_conditional_losses_11793�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_1_layer_call_fn_11798
)__inference_dropout_1_layer_call_fn_11803�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11823
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11857�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
5__inference_batch_normalization_1_layer_call_fn_11870
5__inference_batch_normalization_1_layer_call_fn_11883�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_dense_2_layer_call_and_return_conditional_losses_11894�
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
'__inference_dense_2_layer_call_fn_11903�
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
B__inference_dense_3_layer_call_and_return_conditional_losses_11914�
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
'__inference_dense_3_layer_call_fn_11923�
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
�B�
#__inference_signature_wrapper_10886vgg19_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_block1_conv1_layer_call_and_return_conditional_losses_11934�
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
,__inference_block1_conv1_layer_call_fn_11943�
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
G__inference_block1_conv2_layer_call_and_return_conditional_losses_11954�
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
,__inference_block1_conv2_layer_call_fn_11963�
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
�2�
F__inference_block1_pool_layer_call_and_return_conditional_losses_11968
F__inference_block1_pool_layer_call_and_return_conditional_losses_11973�
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
�2�
+__inference_block1_pool_layer_call_fn_11978
+__inference_block1_pool_layer_call_fn_11983�
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
G__inference_block2_conv1_layer_call_and_return_conditional_losses_11994�
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
,__inference_block2_conv1_layer_call_fn_12003�
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
G__inference_block2_conv2_layer_call_and_return_conditional_losses_12014�
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
,__inference_block2_conv2_layer_call_fn_12023�
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
�2�
F__inference_block2_pool_layer_call_and_return_conditional_losses_12028
F__inference_block2_pool_layer_call_and_return_conditional_losses_12033�
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
�2�
+__inference_block2_pool_layer_call_fn_12038
+__inference_block2_pool_layer_call_fn_12043�
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
G__inference_block3_conv1_layer_call_and_return_conditional_losses_12054�
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
,__inference_block3_conv1_layer_call_fn_12063�
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
G__inference_block3_conv2_layer_call_and_return_conditional_losses_12074�
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
,__inference_block3_conv2_layer_call_fn_12083�
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
G__inference_block3_conv3_layer_call_and_return_conditional_losses_12094�
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
,__inference_block3_conv3_layer_call_fn_12103�
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
G__inference_block3_conv4_layer_call_and_return_conditional_losses_12114�
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
,__inference_block3_conv4_layer_call_fn_12123�
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
�2�
F__inference_block3_pool_layer_call_and_return_conditional_losses_12128
F__inference_block3_pool_layer_call_and_return_conditional_losses_12133�
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
�2�
+__inference_block3_pool_layer_call_fn_12138
+__inference_block3_pool_layer_call_fn_12143�
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
G__inference_block4_conv1_layer_call_and_return_conditional_losses_12154�
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
,__inference_block4_conv1_layer_call_fn_12163�
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
G__inference_block4_conv2_layer_call_and_return_conditional_losses_12174�
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
,__inference_block4_conv2_layer_call_fn_12183�
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
G__inference_block4_conv3_layer_call_and_return_conditional_losses_12194�
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
,__inference_block4_conv3_layer_call_fn_12203�
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
G__inference_block4_conv4_layer_call_and_return_conditional_losses_12214�
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
,__inference_block4_conv4_layer_call_fn_12223�
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
�2�
F__inference_block4_pool_layer_call_and_return_conditional_losses_12228
F__inference_block4_pool_layer_call_and_return_conditional_losses_12233�
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
�2�
+__inference_block4_pool_layer_call_fn_12238
+__inference_block4_pool_layer_call_fn_12243�
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
G__inference_block5_conv1_layer_call_and_return_conditional_losses_12254�
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
,__inference_block5_conv1_layer_call_fn_12263�
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
G__inference_block5_conv2_layer_call_and_return_conditional_losses_12274�
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
,__inference_block5_conv2_layer_call_fn_12283�
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
G__inference_block5_conv3_layer_call_and_return_conditional_losses_12294�
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
,__inference_block5_conv3_layer_call_fn_12303�
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
G__inference_block5_conv4_layer_call_and_return_conditional_losses_12314�
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
,__inference_block5_conv4_layer_call_fn_12323�
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
�2�
F__inference_block5_pool_layer_call_and_return_conditional_losses_12328
F__inference_block5_pool_layer_call_and_return_conditional_losses_12333�
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
�2�
+__inference_block5_pool_layer_call_fn_12338
+__inference_block5_pool_layer_call_fn_12343�
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
 �
__inference__wrapped_model_8697�(IJKLMNOPQRSTUVWXYZ[\]^_`abcdefgh302189>?>�;
4�1
/�,
vgg19_input�����������
� "1�.
,
dense_3!�
dense_3����������
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11823f30215�2
+�(
"�
inputs�����������
p 
� "'�$
�
0�����������
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_11857f23015�2
+�(
"�
inputs�����������
p
� "'�$
�
0�����������
� �
5__inference_batch_normalization_1_layer_call_fn_11870Y30215�2
+�(
"�
inputs�����������
p 
� "�������������
5__inference_batch_normalization_1_layer_call_fn_11883Y23015�2
+�(
"�
inputs�����������
p
� "�������������
G__inference_block1_conv1_layer_call_and_return_conditional_losses_11934pIJ9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������@
� �
,__inference_block1_conv1_layer_call_fn_11943cIJ9�6
/�,
*�'
inputs�����������
� ""������������@�
G__inference_block1_conv2_layer_call_and_return_conditional_losses_11954pKL9�6
/�,
*�'
inputs�����������@
� "/�,
%�"
0�����������@
� �
,__inference_block1_conv2_layer_call_fn_11963cKL9�6
/�,
*�'
inputs�����������@
� ""������������@�
F__inference_block1_pool_layer_call_and_return_conditional_losses_11968�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
F__inference_block1_pool_layer_call_and_return_conditional_losses_11973j9�6
/�,
*�'
inputs�����������@
� "-�*
#� 
0���������pp@
� �
+__inference_block1_pool_layer_call_fn_11978�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
+__inference_block1_pool_layer_call_fn_11983]9�6
/�,
*�'
inputs�����������@
� " ����������pp@�
G__inference_block2_conv1_layer_call_and_return_conditional_losses_11994mMN7�4
-�*
(�%
inputs���������pp@
� ".�+
$�!
0���������pp�
� �
,__inference_block2_conv1_layer_call_fn_12003`MN7�4
-�*
(�%
inputs���������pp@
� "!����������pp��
G__inference_block2_conv2_layer_call_and_return_conditional_losses_12014nOP8�5
.�+
)�&
inputs���������pp�
� ".�+
$�!
0���������pp�
� �
,__inference_block2_conv2_layer_call_fn_12023aOP8�5
.�+
)�&
inputs���������pp�
� "!����������pp��
F__inference_block2_pool_layer_call_and_return_conditional_losses_12028�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
F__inference_block2_pool_layer_call_and_return_conditional_losses_12033j8�5
.�+
)�&
inputs���������pp�
� ".�+
$�!
0���������88�
� �
+__inference_block2_pool_layer_call_fn_12038�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
+__inference_block2_pool_layer_call_fn_12043]8�5
.�+
)�&
inputs���������pp�
� "!����������88��
G__inference_block3_conv1_layer_call_and_return_conditional_losses_12054nQR8�5
.�+
)�&
inputs���������88�
� ".�+
$�!
0���������88�
� �
,__inference_block3_conv1_layer_call_fn_12063aQR8�5
.�+
)�&
inputs���������88�
� "!����������88��
G__inference_block3_conv2_layer_call_and_return_conditional_losses_12074nST8�5
.�+
)�&
inputs���������88�
� ".�+
$�!
0���������88�
� �
,__inference_block3_conv2_layer_call_fn_12083aST8�5
.�+
)�&
inputs���������88�
� "!����������88��
G__inference_block3_conv3_layer_call_and_return_conditional_losses_12094nUV8�5
.�+
)�&
inputs���������88�
� ".�+
$�!
0���������88�
� �
,__inference_block3_conv3_layer_call_fn_12103aUV8�5
.�+
)�&
inputs���������88�
� "!����������88��
G__inference_block3_conv4_layer_call_and_return_conditional_losses_12114nWX8�5
.�+
)�&
inputs���������88�
� ".�+
$�!
0���������88�
� �
,__inference_block3_conv4_layer_call_fn_12123aWX8�5
.�+
)�&
inputs���������88�
� "!����������88��
F__inference_block3_pool_layer_call_and_return_conditional_losses_12128�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
F__inference_block3_pool_layer_call_and_return_conditional_losses_12133j8�5
.�+
)�&
inputs���������88�
� ".�+
$�!
0����������
� �
+__inference_block3_pool_layer_call_fn_12138�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
+__inference_block3_pool_layer_call_fn_12143]8�5
.�+
)�&
inputs���������88�
� "!������������
G__inference_block4_conv1_layer_call_and_return_conditional_losses_12154nYZ8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_block4_conv1_layer_call_fn_12163aYZ8�5
.�+
)�&
inputs����������
� "!������������
G__inference_block4_conv2_layer_call_and_return_conditional_losses_12174n[\8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_block4_conv2_layer_call_fn_12183a[\8�5
.�+
)�&
inputs����������
� "!������������
G__inference_block4_conv3_layer_call_and_return_conditional_losses_12194n]^8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_block4_conv3_layer_call_fn_12203a]^8�5
.�+
)�&
inputs����������
� "!������������
G__inference_block4_conv4_layer_call_and_return_conditional_losses_12214n_`8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_block4_conv4_layer_call_fn_12223a_`8�5
.�+
)�&
inputs����������
� "!������������
F__inference_block4_pool_layer_call_and_return_conditional_losses_12228�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
F__inference_block4_pool_layer_call_and_return_conditional_losses_12233j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
+__inference_block4_pool_layer_call_fn_12238�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
+__inference_block4_pool_layer_call_fn_12243]8�5
.�+
)�&
inputs����������
� "!������������
G__inference_block5_conv1_layer_call_and_return_conditional_losses_12254nab8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_block5_conv1_layer_call_fn_12263aab8�5
.�+
)�&
inputs����������
� "!������������
G__inference_block5_conv2_layer_call_and_return_conditional_losses_12274ncd8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_block5_conv2_layer_call_fn_12283acd8�5
.�+
)�&
inputs����������
� "!������������
G__inference_block5_conv3_layer_call_and_return_conditional_losses_12294nef8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_block5_conv3_layer_call_fn_12303aef8�5
.�+
)�&
inputs����������
� "!������������
G__inference_block5_conv4_layer_call_and_return_conditional_losses_12314ngh8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_block5_conv4_layer_call_fn_12323agh8�5
.�+
)�&
inputs����������
� "!������������
F__inference_block5_pool_layer_call_and_return_conditional_losses_12328�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
F__inference_block5_pool_layer_call_and_return_conditional_losses_12333j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
+__inference_block5_pool_layer_call_fn_12338�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
+__inference_block5_pool_layer_call_fn_12343]8�5
.�+
)�&
inputs����������
� "!������������
B__inference_dense_2_layer_call_and_return_conditional_losses_11894^891�.
'�$
"�
inputs�����������
� "%�"
�
0���������
� |
'__inference_dense_2_layer_call_fn_11903Q891�.
'�$
"�
inputs�����������
� "�����������
B__inference_dense_3_layer_call_and_return_conditional_losses_11914\>?/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
'__inference_dense_3_layer_call_fn_11923O>?/�,
%�"
 �
inputs���������
� "�����������
D__inference_dropout_1_layer_call_and_return_conditional_losses_11781`5�2
+�(
"�
inputs�����������
p 
� "'�$
�
0�����������
� �
D__inference_dropout_1_layer_call_and_return_conditional_losses_11793`5�2
+�(
"�
inputs�����������
p
� "'�$
�
0�����������
� �
)__inference_dropout_1_layer_call_fn_11798S5�2
+�(
"�
inputs�����������
p 
� "�������������
)__inference_dropout_1_layer_call_fn_11803S5�2
+�(
"�
inputs�����������
p
� "�������������
D__inference_flatten_1_layer_call_and_return_conditional_losses_11771c8�5
.�+
)�&
inputs����������
� "'�$
�
0�����������
� �
)__inference_flatten_1_layer_call_fn_11776V8�5
.�+
)�&
inputs����������
� "�������������
G__inference_sequential_1_layer_call_and_return_conditional_losses_10703�(IJKLMNOPQRSTUVWXYZ[\]^_`abcdefgh302189>?F�C
<�9
/�,
vgg19_input�����������
p 

 
� "%�"
�
0���������
� �
G__inference_sequential_1_layer_call_and_return_conditional_losses_10793�(IJKLMNOPQRSTUVWXYZ[\]^_`abcdefgh230189>?F�C
<�9
/�,
vgg19_input�����������
p

 
� "%�"
�
0���������
� �
G__inference_sequential_1_layer_call_and_return_conditional_losses_11040�(IJKLMNOPQRSTUVWXYZ[\]^_`abcdefgh302189>?A�>
7�4
*�'
inputs�����������
p 

 
� "%�"
�
0���������
� �
G__inference_sequential_1_layer_call_and_return_conditional_losses_11215�(IJKLMNOPQRSTUVWXYZ[\]^_`abcdefgh230189>?A�>
7�4
*�'
inputs�����������
p

 
� "%�"
�
0���������
� �
,__inference_sequential_1_layer_call_fn_10219�(IJKLMNOPQRSTUVWXYZ[\]^_`abcdefgh302189>?F�C
<�9
/�,
vgg19_input�����������
p 

 
� "�����������
,__inference_sequential_1_layer_call_fn_10613�(IJKLMNOPQRSTUVWXYZ[\]^_`abcdefgh230189>?F�C
<�9
/�,
vgg19_input�����������
p

 
� "�����������
,__inference_sequential_1_layer_call_fn_11300�(IJKLMNOPQRSTUVWXYZ[\]^_`abcdefgh302189>?A�>
7�4
*�'
inputs�����������
p 

 
� "�����������
,__inference_sequential_1_layer_call_fn_11385�(IJKLMNOPQRSTUVWXYZ[\]^_`abcdefgh230189>?A�>
7�4
*�'
inputs�����������
p

 
� "�����������
#__inference_signature_wrapper_10886�(IJKLMNOPQRSTUVWXYZ[\]^_`abcdefgh302189>?M�J
� 
C�@
>
vgg19_input/�,
vgg19_input�����������"1�.
,
dense_3!�
dense_3����������
@__inference_vgg19_layer_call_and_return_conditional_losses_11506� IJKLMNOPQRSTUVWXYZ[\]^_`abcdefghA�>
7�4
*�'
inputs�����������
p 

 
� ".�+
$�!
0����������
� �
@__inference_vgg19_layer_call_and_return_conditional_losses_11627� IJKLMNOPQRSTUVWXYZ[\]^_`abcdefghA�>
7�4
*�'
inputs�����������
p

 
� ".�+
$�!
0����������
� �
?__inference_vgg19_layer_call_and_return_conditional_losses_9754� IJKLMNOPQRSTUVWXYZ[\]^_`abcdefghB�?
8�5
+�(
input_2�����������
p 

 
� ".�+
$�!
0����������
� �
?__inference_vgg19_layer_call_and_return_conditional_losses_9843� IJKLMNOPQRSTUVWXYZ[\]^_`abcdefghB�?
8�5
+�(
input_2�����������
p

 
� ".�+
$�!
0����������
� �
%__inference_vgg19_layer_call_fn_11696� IJKLMNOPQRSTUVWXYZ[\]^_`abcdefghA�>
7�4
*�'
inputs�����������
p 

 
� "!������������
%__inference_vgg19_layer_call_fn_11765� IJKLMNOPQRSTUVWXYZ[\]^_`abcdefghA�>
7�4
*�'
inputs�����������
p

 
� "!������������
$__inference_vgg19_layer_call_fn_9184� IJKLMNOPQRSTUVWXYZ[\]^_`abcdefghB�?
8�5
+�(
input_2�����������
p 

 
� "!������������
$__inference_vgg19_layer_call_fn_9665� IJKLMNOPQRSTUVWXYZ[\]^_`abcdefghB�?
8�5
+�(
input_2�����������
p

 
� "!�����������