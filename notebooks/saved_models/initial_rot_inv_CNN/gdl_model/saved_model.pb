��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
>
Minimum
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
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
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
y
TensorScatterUpdate
tensor"T
indices"Tindices
updates"T
output"T"	
Ttype"
Tindicestype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.12v2.9.0-18-gd8ce9f9c3018��
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
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

: *
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
: *
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	� *
dtype0
�
rot_equiv_conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namerot_equiv_conv2d_14/bias
�
,rot_equiv_conv2d_14/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_14/bias*
_output_shapes	
:�*
dtype0
�
rot_equiv_conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*+
shared_namerot_equiv_conv2d_14/kernel
�
.rot_equiv_conv2d_14/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_14/kernel*'
_output_shapes
:@�*
dtype0
�
rot_equiv_conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namerot_equiv_conv2d_13/bias
�
,rot_equiv_conv2d_13/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_13/bias*
_output_shapes
:@*
dtype0
�
rot_equiv_conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*+
shared_namerot_equiv_conv2d_13/kernel
�
.rot_equiv_conv2d_13/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_13/kernel*&
_output_shapes
:@@*
dtype0
�
rot_equiv_conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namerot_equiv_conv2d_12/bias
�
,rot_equiv_conv2d_12/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_12/bias*
_output_shapes
:@*
dtype0
�
rot_equiv_conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_namerot_equiv_conv2d_12/kernel
�
.rot_equiv_conv2d_12/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_12/kernel*&
_output_shapes
: @*
dtype0
�
rot_equiv_conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namerot_equiv_conv2d_11/bias
�
,rot_equiv_conv2d_11/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_11/bias*
_output_shapes
: *
dtype0
�
rot_equiv_conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *+
shared_namerot_equiv_conv2d_11/kernel
�
.rot_equiv_conv2d_11/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_11/kernel*&
_output_shapes
:  *
dtype0
�
rot_equiv_conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namerot_equiv_conv2d_10/bias
�
,rot_equiv_conv2d_10/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_10/bias*
_output_shapes
: *
dtype0
�
rot_equiv_conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namerot_equiv_conv2d_10/kernel
�
.rot_equiv_conv2d_10/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_10/kernel*&
_output_shapes
: *
dtype0

NoOpNoOp
�h
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�h
value�hB�h B�h
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
	filt_base
bias
# _self_saveable_object_factories*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'pool
#(_self_saveable_object_factories* 
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
/	filt_base
0bias
#1_self_saveable_object_factories*
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8pool
#9_self_saveable_object_factories* 
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
@	filt_base
Abias
#B_self_saveable_object_factories*
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Ipool
#J_self_saveable_object_factories* 
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Q	filt_base
Rbias
#S_self_saveable_object_factories*
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Zpool
#[_self_saveable_object_factories* 
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
b	filt_base
cbias
#d_self_saveable_object_factories*
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
#k_self_saveable_object_factories* 
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses
#r_self_saveable_object_factories* 
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias
#{_self_saveable_object_factories*
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories*
l
0
1
/2
03
@4
A5
Q6
R7
b8
c9
y10
z11
�12
�13*
l
0
1
/2
03
@4
A5
Q6
R7
b8
c9
y10
z11
�12
�13*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
* 

�serving_default* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUErot_equiv_conv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUErot_equiv_conv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
* 

/0
01*

/0
01*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUErot_equiv_conv2d_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUErot_equiv_conv2d_11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
* 

@0
A1*

@0
A1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUErot_equiv_conv2d_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUErot_equiv_conv2d_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
* 

Q0
R1*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUErot_equiv_conv2d_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUErot_equiv_conv2d_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
* 

b0
c1*

b0
c1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
jd
VARIABLE_VALUErot_equiv_conv2d_14/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUErot_equiv_conv2d_14/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

y0
z1*

y0
z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
'0* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
80* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
I0* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
Z0* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
)serving_default_rot_equiv_conv2d_10_inputPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCall)serving_default_rot_equiv_conv2d_10_inputrot_equiv_conv2d_10/kernelrot_equiv_conv2d_10/biasrot_equiv_conv2d_11/kernelrot_equiv_conv2d_11/biasrot_equiv_conv2d_12/kernelrot_equiv_conv2d_12/biasrot_equiv_conv2d_13/kernelrot_equiv_conv2d_13/biasrot_equiv_conv2d_14/kernelrot_equiv_conv2d_14/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_631159
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.rot_equiv_conv2d_10/kernel/Read/ReadVariableOp,rot_equiv_conv2d_10/bias/Read/ReadVariableOp.rot_equiv_conv2d_11/kernel/Read/ReadVariableOp,rot_equiv_conv2d_11/bias/Read/ReadVariableOp.rot_equiv_conv2d_12/kernel/Read/ReadVariableOp,rot_equiv_conv2d_12/bias/Read/ReadVariableOp.rot_equiv_conv2d_13/kernel/Read/ReadVariableOp,rot_equiv_conv2d_13/bias/Read/ReadVariableOp.rot_equiv_conv2d_14/kernel/Read/ReadVariableOp,rot_equiv_conv2d_14/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8� *(
f#R!
__inference__traced_save_631538
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerot_equiv_conv2d_10/kernelrot_equiv_conv2d_10/biasrot_equiv_conv2d_11/kernelrot_equiv_conv2d_11/biasrot_equiv_conv2d_12/kernelrot_equiv_conv2d_12/biasrot_equiv_conv2d_13/kernelrot_equiv_conv2d_13/biasrot_equiv_conv2d_14/kernelrot_equiv_conv2d_14/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biastotal_1count_1totalcount*
Tin
2*
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
GPU2*0J 8� *+
f&R$
"__inference__traced_restore_631602��
�
�
4__inference_rot_equiv_conv2d_10_layer_call_fn_494838

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_conv2d_10_layer_call_and_return_conditional_losses_494831`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 }
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
K
/__inference_rot_inv_pool_2_layer_call_fn_494854

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_rot_inv_pool_2_layer_call_and_return_conditional_losses_494849i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�

�
C__inference_dense_6_layer_call_and_return_conditional_losses_630810

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_631332

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�-
�
__inference__traced_save_631538
file_prefix9
5savev2_rot_equiv_conv2d_10_kernel_read_readvariableop7
3savev2_rot_equiv_conv2d_10_bias_read_readvariableop9
5savev2_rot_equiv_conv2d_11_kernel_read_readvariableop7
3savev2_rot_equiv_conv2d_11_bias_read_readvariableop9
5savev2_rot_equiv_conv2d_12_kernel_read_readvariableop7
3savev2_rot_equiv_conv2d_12_bias_read_readvariableop9
5savev2_rot_equiv_conv2d_13_kernel_read_readvariableop7
3savev2_rot_equiv_conv2d_13_bias_read_readvariableop9
5savev2_rot_equiv_conv2d_14_kernel_read_readvariableop7
3savev2_rot_equiv_conv2d_14_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_rot_equiv_conv2d_10_kernel_read_readvariableop3savev2_rot_equiv_conv2d_10_bias_read_readvariableop5savev2_rot_equiv_conv2d_11_kernel_read_readvariableop3savev2_rot_equiv_conv2d_11_bias_read_readvariableop5savev2_rot_equiv_conv2d_12_kernel_read_readvariableop3savev2_rot_equiv_conv2d_12_bias_read_readvariableop5savev2_rot_equiv_conv2d_13_kernel_read_readvariableop3savev2_rot_equiv_conv2d_13_bias_read_readvariableop5savev2_rot_equiv_conv2d_14_kernel_read_readvariableop3savev2_rot_equiv_conv2d_14_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : : @:@:@@:@:@�:�:	� : : :: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-	)
'
_output_shapes
:@�:!


_output_shapes	
:�:%!

_output_shapes
:	� : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
h
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_631431

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�6
j
N__inference_rot_equiv_pool2d_8_layer_call_and_return_conditional_losses_492956

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:����������� �
max_pooling2d_12/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:���������GG *
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:����������� �
max_pooling2d_12/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:���������GG *
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:����������� �
max_pooling2d_12/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:���������GG *
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:����������� �
max_pooling2d_12/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:���������GG *
ksize
*
paddingVALID*
strides
�
stackPack!max_pooling2d_12/MaxPool:output:0#max_pooling2d_12/MaxPool_1:output:0#max_pooling2d_12/MaxPool_2:output:0#max_pooling2d_12/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:���������GG *
axis���������b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:���������GG "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������� :] Y
5
_output_shapes#
!:����������� 
 
_user_specified_nameinputs
�C
�
O__inference_rot_equiv_conv2d_14_layer_call_and_return_conditional_losses_493643

inputs>
#convolution_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�convolution_1/ReadVariableOp�convolution_2/ReadVariableOp�convolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*4
_output_shapes"
 :����������*
axis���������[
ReluRelustack:output:0*
T0*4
_output_shapes"
 :����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
�
)__inference_restored_function_body_630666

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*3
_output_shapes!
:���������EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_conv2d_11_layer_call_and_return_conditional_losses_493720{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������EE `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������GG : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������GG 
 
_user_specified_nameinputs
�>
�	
H__inference_sequential_3_layer_call_and_return_conditional_losses_631274

inputs4
rot_equiv_conv2d_10_631228: (
rot_equiv_conv2d_10_631230: 4
rot_equiv_conv2d_11_631234:  (
rot_equiv_conv2d_11_631236: 4
rot_equiv_conv2d_12_631240: @(
rot_equiv_conv2d_12_631242:@4
rot_equiv_conv2d_13_631246:@@(
rot_equiv_conv2d_13_631248:@5
rot_equiv_conv2d_14_631252:@�)
rot_equiv_conv2d_14_631254:	�9
&dense_6_matmul_readvariableop_resource:	� 5
'dense_6_biasadd_readvariableop_resource: 8
&dense_7_matmul_readvariableop_resource: 5
'dense_7_biasadd_readvariableop_resource:
identity��dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�+rot_equiv_conv2d_10/StatefulPartitionedCall�+rot_equiv_conv2d_11/StatefulPartitionedCall�+rot_equiv_conv2d_12/StatefulPartitionedCall�+rot_equiv_conv2d_13/StatefulPartitionedCall�+rot_equiv_conv2d_14/StatefulPartitionedCall�
+rot_equiv_conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsrot_equiv_conv2d_10_631228rot_equiv_conv2d_10_631230*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630646�
"rot_equiv_pool2d_8/PartitionedCallPartitionedCall4rot_equiv_conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630656�
+rot_equiv_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_8/PartitionedCall:output:0rot_equiv_conv2d_11_631234rot_equiv_conv2d_11_631236*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630666�
"rot_equiv_pool2d_9/PartitionedCallPartitionedCall4rot_equiv_conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630676�
+rot_equiv_conv2d_12/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_9/PartitionedCall:output:0rot_equiv_conv2d_12_631240rot_equiv_conv2d_12_631242*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630686�
#rot_equiv_pool2d_10/PartitionedCallPartitionedCall4rot_equiv_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630696�
+rot_equiv_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_10/PartitionedCall:output:0rot_equiv_conv2d_13_631246rot_equiv_conv2d_13_631248*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630706�
#rot_equiv_pool2d_11/PartitionedCallPartitionedCall4rot_equiv_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630716�
+rot_equiv_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_11/PartitionedCall:output:0rot_equiv_conv2d_14_631252rot_equiv_conv2d_14_631254*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630726�
rot_inv_pool_2/PartitionedCallPartitionedCall4rot_equiv_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630736`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten_3/ReshapeReshape'rot_inv_pool_2/PartitionedCall:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:�����������
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp,^rot_equiv_conv2d_10/StatefulPartitionedCall,^rot_equiv_conv2d_11/StatefulPartitionedCall,^rot_equiv_conv2d_12/StatefulPartitionedCall,^rot_equiv_conv2d_13/StatefulPartitionedCall,^rot_equiv_conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2Z
+rot_equiv_conv2d_10/StatefulPartitionedCall+rot_equiv_conv2d_10/StatefulPartitionedCall2Z
+rot_equiv_conv2d_11/StatefulPartitionedCall+rot_equiv_conv2d_11/StatefulPartitionedCall2Z
+rot_equiv_conv2d_12/StatefulPartitionedCall+rot_equiv_conv2d_12/StatefulPartitionedCall2Z
+rot_equiv_conv2d_13/StatefulPartitionedCall+rot_equiv_conv2d_13/StatefulPartitionedCall2Z
+rot_equiv_conv2d_14/StatefulPartitionedCall+rot_equiv_conv2d_14/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
O
3__inference_rot_equiv_pool2d_9_layer_call_fn_494380

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_rot_equiv_pool2d_9_layer_call_and_return_conditional_losses_494375l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"" "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������EE :[ W
3
_output_shapes!
:���������EE 
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_631159
rot_equiv_conv2d_10_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�
	unknown_9:	� 

unknown_10: 

unknown_11: 

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_630754o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
1
_output_shapes
:�����������
3
_user_specified_namerot_equiv_conv2d_10_input
�H
�
O__inference_rot_equiv_conv2d_10_layer_call_and_return_conditional_losses_494831

inputs=
#convolution_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
convolutionConv2Dinputs"convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides

Rank/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:t
TensorScatterUpdate/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       l
TensorScatterUpdate/updatesConst*
_output_shapes
:*
dtype0*
valueB"       �
TensorScatterUpdateTensorScatterUpdaterange:output:0$TensorScatterUpdate/indices:output:0$TensorScatterUpdate/updates:output:0*
T0*
Tindices0*
_output_shapes
:z
ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:x
	ReverseV2	ReverseV2ReadVariableOp:value:0ReverseV2/axis:output:0*
T0*&
_output_shapes
: y
	transpose	TransposeReverseV2:output:0TensorScatterUpdate:output:0*
T0*&
_output_shapes
: �
convolution_1Conv2Dinputstranspose:y:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :m
range_1Rangerange_1/start:output:0Rank_2:output:0range_1/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_1/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      n
TensorScatterUpdate_1/updatesConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorScatterUpdate_1TensorScatterUpdaterange_1:output:0&TensorScatterUpdate_1/indices:output:0&TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:�
transpose_1	Transposeconvolution_1:output:0TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:����������� H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:�
ReverseV2_1	ReverseV2transpose_1:y:0ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:����������� �
Rank_4/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_4Const*
_output_shapes
: *
dtype0*
value	B :|
ReadVariableOp_1ReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_5Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_2/axisConst*
_output_shapes
:*
dtype0*
valueB: ~
ReverseV2_2	ReverseV2ReadVariableOp_1:value:0ReverseV2_2/axis:output:0*
T0*&
_output_shapes
: H
Rank_6Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_3/axisConst*
_output_shapes
:*
dtype0*
valueB:z
ReverseV2_3	ReverseV2ReverseV2_2:output:0ReverseV2_3/axis:output:0*
T0*&
_output_shapes
: �
convolution_2Conv2DinputsReverseV2_3:output:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
H
Rank_7Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_8Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_4/axisConst*
_output_shapes
:*
dtype0*
valueB:�
ReverseV2_4	ReverseV2convolution_2:output:0ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:����������� H
Rank_9Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:�
ReverseV2_5	ReverseV2ReverseV2_4:output:0ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:����������� �
Rank_10/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0I
Rank_10Const*
_output_shapes
: *
dtype0*
value	B :O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
range_2Rangerange_2/start:output:0Rank_10:output:0range_2/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_2/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       n
TensorScatterUpdate_2/updatesConst*
_output_shapes
:*
dtype0*
valueB"       �
TensorScatterUpdate_2TensorScatterUpdaterange_2:output:0&TensorScatterUpdate_2/indices:output:0&TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:|
ReadVariableOp_2ReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
transpose_2	TransposeReadVariableOp_2:value:0TensorScatterUpdate_2:output:0*
T0*&
_output_shapes
: I
Rank_11Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_6/axisConst*
_output_shapes
:*
dtype0*
valueB:u
ReverseV2_6	ReverseV2transpose_2:y:0ReverseV2_6/axis:output:0*
T0*&
_output_shapes
: �
convolution_3Conv2DinputsReverseV2_6:output:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
I
Rank_12Const*
_output_shapes
: *
dtype0*
value	B :O
range_3/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
range_3Rangerange_3/start:output:0Rank_12:output:0range_3/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_3/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      n
TensorScatterUpdate_3/updatesConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorScatterUpdate_3TensorScatterUpdaterange_3:output:0&TensorScatterUpdate_3/indices:output:0&TensorScatterUpdate_3/updates:output:0*
T0*
Tindices0*
_output_shapes
:I
Rank_13Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_7/axisConst*
_output_shapes
:*
dtype0*
valueB:�
ReverseV2_7	ReverseV2convolution_3:output:0ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:����������� �
transpose_3	TransposeReverseV2_7:output:0TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:����������� �
stackPackconvolution:output:0ReverseV2_1:output:0ReverseV2_5:output:0transpose_3:y:0*
N*
T0*5
_output_shapes#
!:����������� *
axis���������\
ReluRelustack:output:0*
T0*5
_output_shapes#
!:����������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:����������� �
NoOpNoOp^BiasAdd/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^convolution/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_228
convolution/ReadVariableOpconvolution/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�C
�
O__inference_rot_equiv_conv2d_11_layer_call_and_return_conditional_losses_493720

inputs=
#convolution_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�convolution_1/ReadVariableOp�convolution_2/ReadVariableOp�convolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������GG �
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0�
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������EE *
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������GG �
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0�
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������EE *
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������GG �
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0�
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������EE *
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������GG �
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0�
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������EE *
paddingVALID*
strides
�
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:���������EE *
axis���������Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:���������EE r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������EE �
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������EE "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������GG : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:���������GG 
 
_user_specified_nameinputs
�6
k
O__inference_rot_equiv_pool2d_10_layer_call_and_return_conditional_losses_494594

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������  @�
max_pooling2d_14/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������  @�
max_pooling2d_14/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������  @�
max_pooling2d_14/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������  @�
max_pooling2d_14/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
stackPack!max_pooling2d_14/MaxPool:output:0#max_pooling2d_14/MaxPool_1:output:0#max_pooling2d_14/MaxPool_2:output:0#max_pooling2d_14/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:���������@*
axis���������b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @:[ W
3
_output_shapes!
:���������  @
 
_user_specified_nameinputs
�6
k
O__inference_rot_equiv_pool2d_10_layer_call_and_return_conditional_losses_492187

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������  @�
max_pooling2d_14/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������  @�
max_pooling2d_14/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������  @�
max_pooling2d_14/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������  @�
max_pooling2d_14/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
stackPack!max_pooling2d_14/MaxPool:output:0#max_pooling2d_14/MaxPool_1:output:0#max_pooling2d_14/MaxPool_2:output:0#max_pooling2d_14/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:���������@*
axis���������b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @:[ W
3
_output_shapes!
:���������  @
 
_user_specified_nameinputs
�
�
-__inference_sequential_3_layer_call_fn_631225

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�
	unknown_9:	� 

unknown_10: 

unknown_11: 

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_630970o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_631368

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�6
k
O__inference_rot_equiv_pool2d_11_layer_call_and_return_conditional_losses_494752

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
max_pooling2d_15/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
max_pooling2d_15/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
max_pooling2d_15/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
max_pooling2d_15/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
stackPack!max_pooling2d_15/MaxPool:output:0#max_pooling2d_15/MaxPool_1:output:0#max_pooling2d_15/MaxPool_2:output:0#max_pooling2d_15/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:���������@*
axis���������b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_12_layer_call_fn_631426

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
GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_631332�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�6
k
O__inference_rot_equiv_pool2d_11_layer_call_and_return_conditional_losses_494528

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
max_pooling2d_15/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
max_pooling2d_15/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
max_pooling2d_15/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
max_pooling2d_15/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
stackPack!max_pooling2d_15/MaxPool:output:0#max_pooling2d_15/MaxPool_1:output:0#max_pooling2d_15/MaxPool_2:output:0#max_pooling2d_15/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:���������@*
axis���������b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
�
4__inference_rot_equiv_conv2d_14_layer_call_fn_493650

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_conv2d_14_layer_call_and_return_conditional_losses_493643`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 |
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_631441

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
P
4__inference_rot_equiv_pool2d_11_layer_call_fn_494533

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_pool2d_11_layer_call_and_return_conditional_losses_494528l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
E
)__inference_restored_function_body_630676

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:���������"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_rot_equiv_pool2d_9_layer_call_and_return_conditional_losses_491736l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������"" "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������EE :[ W
3
_output_shapes!
:���������EE 
 
_user_specified_nameinputs
�
E
)__inference_restored_function_body_630716

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_pool2d_11_layer_call_and_return_conditional_losses_494752l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�C
�
O__inference_rot_equiv_conv2d_13_layer_call_and_return_conditional_losses_494669

inputs=
#convolution_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�convolution_1/ReadVariableOp�convolution_2/ReadVariableOp�convolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:���������@*
axis���������Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
�
(__inference_dense_6_layer_call_fn_631391

inputs
unknown:	� 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_630810o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_restored_function_body_630696

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_pool2d_10_layer_call_and_return_conditional_losses_492187l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @:[ W
3
_output_shapes!
:���������  @
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_15_layer_call_fn_631456

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
GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_631368�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�J
�
"__inference__traced_restore_631602
file_prefixE
+assignvariableop_rot_equiv_conv2d_10_kernel: 9
+assignvariableop_1_rot_equiv_conv2d_10_bias: G
-assignvariableop_2_rot_equiv_conv2d_11_kernel:  9
+assignvariableop_3_rot_equiv_conv2d_11_bias: G
-assignvariableop_4_rot_equiv_conv2d_12_kernel: @9
+assignvariableop_5_rot_equiv_conv2d_12_bias:@G
-assignvariableop_6_rot_equiv_conv2d_13_kernel:@@9
+assignvariableop_7_rot_equiv_conv2d_13_bias:@H
-assignvariableop_8_rot_equiv_conv2d_14_kernel:@�:
+assignvariableop_9_rot_equiv_conv2d_14_bias:	�5
"assignvariableop_10_dense_6_kernel:	� .
 assignvariableop_11_dense_6_bias: 4
"assignvariableop_12_dense_7_kernel: .
 assignvariableop_13_dense_7_bias:%
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp+assignvariableop_rot_equiv_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_rot_equiv_conv2d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp-assignvariableop_2_rot_equiv_conv2d_11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_rot_equiv_conv2d_11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp-assignvariableop_4_rot_equiv_conv2d_12_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_rot_equiv_conv2d_12_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_rot_equiv_conv2d_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_rot_equiv_conv2d_13_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_rot_equiv_conv2d_14_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp+assignvariableop_9_rot_equiv_conv2d_14_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
-__inference_sequential_3_layer_call_fn_631192

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�
	unknown_9:	� 

unknown_10: 

unknown_11: 

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_630833o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
E
)__inference_restored_function_body_630656

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*3
_output_shapes!
:���������GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_rot_equiv_pool2d_8_layer_call_and_return_conditional_losses_495185l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������GG "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������� :] Y
5
_output_shapes#
!:����������� 
 
_user_specified_nameinputs
�
E
)__inference_restored_function_body_630736

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_rot_inv_pool_2_layer_call_and_return_conditional_losses_494314i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
F
*__inference_flatten_3_layer_call_fn_631376

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_630797a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_restored_function_body_630706

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_conv2d_13_layer_call_and_return_conditional_losses_495124{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_13_layer_call_fn_631436

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
GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_631344�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_631451

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
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
4__inference_rot_equiv_conv2d_13_layer_call_fn_494676

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_conv2d_13_layer_call_and_return_conditional_losses_494669`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 {
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�:
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_631079
rot_equiv_conv2d_10_input4
rot_equiv_conv2d_10_631037: (
rot_equiv_conv2d_10_631039: 4
rot_equiv_conv2d_11_631043:  (
rot_equiv_conv2d_11_631045: 4
rot_equiv_conv2d_12_631049: @(
rot_equiv_conv2d_12_631051:@4
rot_equiv_conv2d_13_631055:@@(
rot_equiv_conv2d_13_631057:@5
rot_equiv_conv2d_14_631061:@�)
rot_equiv_conv2d_14_631063:	�!
dense_6_631068:	� 
dense_6_631070:  
dense_7_631073: 
dense_7_631075:
identity��dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�+rot_equiv_conv2d_10/StatefulPartitionedCall�+rot_equiv_conv2d_11/StatefulPartitionedCall�+rot_equiv_conv2d_12/StatefulPartitionedCall�+rot_equiv_conv2d_13/StatefulPartitionedCall�+rot_equiv_conv2d_14/StatefulPartitionedCall�
+rot_equiv_conv2d_10/StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_10_inputrot_equiv_conv2d_10_631037rot_equiv_conv2d_10_631039*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630646�
"rot_equiv_pool2d_8/PartitionedCallPartitionedCall4rot_equiv_conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630656�
+rot_equiv_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_8/PartitionedCall:output:0rot_equiv_conv2d_11_631043rot_equiv_conv2d_11_631045*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630666�
"rot_equiv_pool2d_9/PartitionedCallPartitionedCall4rot_equiv_conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630676�
+rot_equiv_conv2d_12/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_9/PartitionedCall:output:0rot_equiv_conv2d_12_631049rot_equiv_conv2d_12_631051*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630686�
#rot_equiv_pool2d_10/PartitionedCallPartitionedCall4rot_equiv_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630696�
+rot_equiv_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_10/PartitionedCall:output:0rot_equiv_conv2d_13_631055rot_equiv_conv2d_13_631057*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630706�
#rot_equiv_pool2d_11/PartitionedCallPartitionedCall4rot_equiv_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630716�
+rot_equiv_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_11/PartitionedCall:output:0rot_equiv_conv2d_14_631061rot_equiv_conv2d_14_631063*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630726�
rot_inv_pool_2/PartitionedCallPartitionedCall4rot_equiv_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630736�
flatten_3/PartitionedCallPartitionedCall'rot_inv_pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_630797�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_631068dense_6_631070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_630810�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_631073dense_7_631075*
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
GPU2*0J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_630826w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^rot_equiv_conv2d_10/StatefulPartitionedCall,^rot_equiv_conv2d_11/StatefulPartitionedCall,^rot_equiv_conv2d_12/StatefulPartitionedCall,^rot_equiv_conv2d_13/StatefulPartitionedCall,^rot_equiv_conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+rot_equiv_conv2d_10/StatefulPartitionedCall+rot_equiv_conv2d_10/StatefulPartitionedCall2Z
+rot_equiv_conv2d_11/StatefulPartitionedCall+rot_equiv_conv2d_11/StatefulPartitionedCall2Z
+rot_equiv_conv2d_12/StatefulPartitionedCall+rot_equiv_conv2d_12/StatefulPartitionedCall2Z
+rot_equiv_conv2d_13/StatefulPartitionedCall+rot_equiv_conv2d_13/StatefulPartitionedCall2Z
+rot_equiv_conv2d_14/StatefulPartitionedCall+rot_equiv_conv2d_14/StatefulPartitionedCall:l h
1
_output_shapes
:�����������
3
_user_specified_namerot_equiv_conv2d_10_input
�
�
)__inference_restored_function_body_630646

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*5
_output_shapes#
!:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_conv2d_10_layer_call_and_return_conditional_losses_491815}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:����������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�C
�
O__inference_rot_equiv_conv2d_12_layer_call_and_return_conditional_losses_492119

inputs=
#convolution_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�convolution_1/ReadVariableOp�convolution_2/ReadVariableOp�convolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������"" �
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������"" �
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������"" �
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������"" �
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingVALID*
strides
�
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:���������  @*
axis���������Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:���������  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������  @�
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������"" : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:���������"" 
 
_user_specified_nameinputs
�6
j
N__inference_rot_equiv_pool2d_9_layer_call_and_return_conditional_losses_494375

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������EE �
max_pooling2d_13/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:���������"" *
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������EE �
max_pooling2d_13/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:���������"" *
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������EE �
max_pooling2d_13/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:���������"" *
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������EE �
max_pooling2d_13/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:���������"" *
ksize
*
paddingVALID*
strides
�
stackPack!max_pooling2d_13/MaxPool:output:0#max_pooling2d_13/MaxPool_1:output:0#max_pooling2d_13/MaxPool_2:output:0#max_pooling2d_13/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:���������"" *
axis���������b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:���������"" "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������EE :[ W
3
_output_shapes!
:���������EE 
 
_user_specified_nameinputs
�C
�
O__inference_rot_equiv_conv2d_13_layer_call_and_return_conditional_losses_495124

inputs=
#convolution_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�convolution_1/ReadVariableOp�convolution_2/ReadVariableOp�convolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:���������@*
axis���������Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�C
�
O__inference_rot_equiv_conv2d_12_layer_call_and_return_conditional_losses_492049

inputs=
#convolution_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�convolution_1/ReadVariableOp�convolution_2/ReadVariableOp�convolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������"" �
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������"" �
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������"" �
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������"" �
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingVALID*
strides
�
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:���������  @*
axis���������Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:���������  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������  @�
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������"" : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:���������"" 
 
_user_specified_nameinputs
�9
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_630970

inputs4
rot_equiv_conv2d_10_630928: (
rot_equiv_conv2d_10_630930: 4
rot_equiv_conv2d_11_630934:  (
rot_equiv_conv2d_11_630936: 4
rot_equiv_conv2d_12_630940: @(
rot_equiv_conv2d_12_630942:@4
rot_equiv_conv2d_13_630946:@@(
rot_equiv_conv2d_13_630948:@5
rot_equiv_conv2d_14_630952:@�)
rot_equiv_conv2d_14_630954:	�!
dense_6_630959:	� 
dense_6_630961:  
dense_7_630964: 
dense_7_630966:
identity��dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�+rot_equiv_conv2d_10/StatefulPartitionedCall�+rot_equiv_conv2d_11/StatefulPartitionedCall�+rot_equiv_conv2d_12/StatefulPartitionedCall�+rot_equiv_conv2d_13/StatefulPartitionedCall�+rot_equiv_conv2d_14/StatefulPartitionedCall�
+rot_equiv_conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsrot_equiv_conv2d_10_630928rot_equiv_conv2d_10_630930*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630646�
"rot_equiv_pool2d_8/PartitionedCallPartitionedCall4rot_equiv_conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630656�
+rot_equiv_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_8/PartitionedCall:output:0rot_equiv_conv2d_11_630934rot_equiv_conv2d_11_630936*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630666�
"rot_equiv_pool2d_9/PartitionedCallPartitionedCall4rot_equiv_conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630676�
+rot_equiv_conv2d_12/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_9/PartitionedCall:output:0rot_equiv_conv2d_12_630940rot_equiv_conv2d_12_630942*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630686�
#rot_equiv_pool2d_10/PartitionedCallPartitionedCall4rot_equiv_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630696�
+rot_equiv_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_10/PartitionedCall:output:0rot_equiv_conv2d_13_630946rot_equiv_conv2d_13_630948*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630706�
#rot_equiv_pool2d_11/PartitionedCallPartitionedCall4rot_equiv_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630716�
+rot_equiv_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_11/PartitionedCall:output:0rot_equiv_conv2d_14_630952rot_equiv_conv2d_14_630954*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630726�
rot_inv_pool_2/PartitionedCallPartitionedCall4rot_equiv_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630736�
flatten_3/PartitionedCallPartitionedCall'rot_inv_pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_630797�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_630959dense_6_630961*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_630810�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_630964dense_7_630966*
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
GPU2*0J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_630826w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^rot_equiv_conv2d_10/StatefulPartitionedCall,^rot_equiv_conv2d_11/StatefulPartitionedCall,^rot_equiv_conv2d_12/StatefulPartitionedCall,^rot_equiv_conv2d_13/StatefulPartitionedCall,^rot_equiv_conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+rot_equiv_conv2d_10/StatefulPartitionedCall+rot_equiv_conv2d_10/StatefulPartitionedCall2Z
+rot_equiv_conv2d_11/StatefulPartitionedCall+rot_equiv_conv2d_11/StatefulPartitionedCall2Z
+rot_equiv_conv2d_12/StatefulPartitionedCall+rot_equiv_conv2d_12/StatefulPartitionedCall2Z
+rot_equiv_conv2d_13/StatefulPartitionedCall+rot_equiv_conv2d_13/StatefulPartitionedCall2Z
+rot_equiv_conv2d_14/StatefulPartitionedCall+rot_equiv_conv2d_14/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
-__inference_sequential_3_layer_call_fn_630864
rot_equiv_conv2d_10_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�
	unknown_9:	� 

unknown_10: 

unknown_11: 

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_630833o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
1
_output_shapes
:�����������
3
_user_specified_namerot_equiv_conv2d_10_input
�C
�
O__inference_rot_equiv_conv2d_11_layer_call_and_return_conditional_losses_491634

inputs=
#convolution_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�convolution_1/ReadVariableOp�convolution_2/ReadVariableOp�convolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������GG �
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0�
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������EE *
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������GG �
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0�
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������EE *
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������GG �
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0�
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������EE *
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������GG �
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0�
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������EE *
paddingVALID*
strides
�
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:���������EE *
axis���������Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:���������EE r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������EE �
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������EE "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������GG : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:���������GG 
 
_user_specified_nameinputs
�

�
C__inference_dense_6_layer_call_and_return_conditional_losses_631402

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_7_layer_call_fn_631411

inputs
unknown: 
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
GPU2*0J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_630826o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�:
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_631124
rot_equiv_conv2d_10_input4
rot_equiv_conv2d_10_631082: (
rot_equiv_conv2d_10_631084: 4
rot_equiv_conv2d_11_631088:  (
rot_equiv_conv2d_11_631090: 4
rot_equiv_conv2d_12_631094: @(
rot_equiv_conv2d_12_631096:@4
rot_equiv_conv2d_13_631100:@@(
rot_equiv_conv2d_13_631102:@5
rot_equiv_conv2d_14_631106:@�)
rot_equiv_conv2d_14_631108:	�!
dense_6_631113:	� 
dense_6_631115:  
dense_7_631118: 
dense_7_631120:
identity��dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�+rot_equiv_conv2d_10/StatefulPartitionedCall�+rot_equiv_conv2d_11/StatefulPartitionedCall�+rot_equiv_conv2d_12/StatefulPartitionedCall�+rot_equiv_conv2d_13/StatefulPartitionedCall�+rot_equiv_conv2d_14/StatefulPartitionedCall�
+rot_equiv_conv2d_10/StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_10_inputrot_equiv_conv2d_10_631082rot_equiv_conv2d_10_631084*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630646�
"rot_equiv_pool2d_8/PartitionedCallPartitionedCall4rot_equiv_conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630656�
+rot_equiv_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_8/PartitionedCall:output:0rot_equiv_conv2d_11_631088rot_equiv_conv2d_11_631090*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630666�
"rot_equiv_pool2d_9/PartitionedCallPartitionedCall4rot_equiv_conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630676�
+rot_equiv_conv2d_12/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_9/PartitionedCall:output:0rot_equiv_conv2d_12_631094rot_equiv_conv2d_12_631096*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630686�
#rot_equiv_pool2d_10/PartitionedCallPartitionedCall4rot_equiv_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630696�
+rot_equiv_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_10/PartitionedCall:output:0rot_equiv_conv2d_13_631100rot_equiv_conv2d_13_631102*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630706�
#rot_equiv_pool2d_11/PartitionedCallPartitionedCall4rot_equiv_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630716�
+rot_equiv_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_11/PartitionedCall:output:0rot_equiv_conv2d_14_631106rot_equiv_conv2d_14_631108*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630726�
rot_inv_pool_2/PartitionedCallPartitionedCall4rot_equiv_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630736�
flatten_3/PartitionedCallPartitionedCall'rot_inv_pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_630797�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_631113dense_6_631115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_630810�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_631118dense_7_631120*
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
GPU2*0J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_630826w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^rot_equiv_conv2d_10/StatefulPartitionedCall,^rot_equiv_conv2d_11/StatefulPartitionedCall,^rot_equiv_conv2d_12/StatefulPartitionedCall,^rot_equiv_conv2d_13/StatefulPartitionedCall,^rot_equiv_conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+rot_equiv_conv2d_10/StatefulPartitionedCall+rot_equiv_conv2d_10/StatefulPartitionedCall2Z
+rot_equiv_conv2d_11/StatefulPartitionedCall+rot_equiv_conv2d_11/StatefulPartitionedCall2Z
+rot_equiv_conv2d_12/StatefulPartitionedCall+rot_equiv_conv2d_12/StatefulPartitionedCall2Z
+rot_equiv_conv2d_13/StatefulPartitionedCall+rot_equiv_conv2d_13/StatefulPartitionedCall2Z
+rot_equiv_conv2d_14/StatefulPartitionedCall+rot_equiv_conv2d_14/StatefulPartitionedCall:l h
1
_output_shapes
:�����������
3
_user_specified_namerot_equiv_conv2d_10_input
�
M
1__inference_max_pooling2d_14_layer_call_fn_631446

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
GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_631356�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_630797

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�I
�
!__inference__wrapped_model_630754
rot_equiv_conv2d_10_inputA
'sequential_3_rot_equiv_conv2d_10_630647: 5
'sequential_3_rot_equiv_conv2d_10_630649: A
'sequential_3_rot_equiv_conv2d_11_630667:  5
'sequential_3_rot_equiv_conv2d_11_630669: A
'sequential_3_rot_equiv_conv2d_12_630687: @5
'sequential_3_rot_equiv_conv2d_12_630689:@A
'sequential_3_rot_equiv_conv2d_13_630707:@@5
'sequential_3_rot_equiv_conv2d_13_630709:@B
'sequential_3_rot_equiv_conv2d_14_630727:@�6
'sequential_3_rot_equiv_conv2d_14_630729:	�F
3sequential_3_dense_6_matmul_readvariableop_resource:	� B
4sequential_3_dense_6_biasadd_readvariableop_resource: E
3sequential_3_dense_7_matmul_readvariableop_resource: B
4sequential_3_dense_7_biasadd_readvariableop_resource:
identity��+sequential_3/dense_6/BiasAdd/ReadVariableOp�*sequential_3/dense_6/MatMul/ReadVariableOp�+sequential_3/dense_7/BiasAdd/ReadVariableOp�*sequential_3/dense_7/MatMul/ReadVariableOp�8sequential_3/rot_equiv_conv2d_10/StatefulPartitionedCall�8sequential_3/rot_equiv_conv2d_11/StatefulPartitionedCall�8sequential_3/rot_equiv_conv2d_12/StatefulPartitionedCall�8sequential_3/rot_equiv_conv2d_13/StatefulPartitionedCall�8sequential_3/rot_equiv_conv2d_14/StatefulPartitionedCall�
8sequential_3/rot_equiv_conv2d_10/StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_10_input'sequential_3_rot_equiv_conv2d_10_630647'sequential_3_rot_equiv_conv2d_10_630649*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630646�
/sequential_3/rot_equiv_pool2d_8/PartitionedCallPartitionedCallAsequential_3/rot_equiv_conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630656�
8sequential_3/rot_equiv_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall8sequential_3/rot_equiv_pool2d_8/PartitionedCall:output:0'sequential_3_rot_equiv_conv2d_11_630667'sequential_3_rot_equiv_conv2d_11_630669*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630666�
/sequential_3/rot_equiv_pool2d_9/PartitionedCallPartitionedCallAsequential_3/rot_equiv_conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630676�
8sequential_3/rot_equiv_conv2d_12/StatefulPartitionedCallStatefulPartitionedCall8sequential_3/rot_equiv_pool2d_9/PartitionedCall:output:0'sequential_3_rot_equiv_conv2d_12_630687'sequential_3_rot_equiv_conv2d_12_630689*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630686�
0sequential_3/rot_equiv_pool2d_10/PartitionedCallPartitionedCallAsequential_3/rot_equiv_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630696�
8sequential_3/rot_equiv_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall9sequential_3/rot_equiv_pool2d_10/PartitionedCall:output:0'sequential_3_rot_equiv_conv2d_13_630707'sequential_3_rot_equiv_conv2d_13_630709*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630706�
0sequential_3/rot_equiv_pool2d_11/PartitionedCallPartitionedCallAsequential_3/rot_equiv_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630716�
8sequential_3/rot_equiv_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall9sequential_3/rot_equiv_pool2d_11/PartitionedCall:output:0'sequential_3_rot_equiv_conv2d_14_630727'sequential_3_rot_equiv_conv2d_14_630729*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630726�
+sequential_3/rot_inv_pool_2/PartitionedCallPartitionedCallAsequential_3/rot_equiv_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630736m
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
sequential_3/flatten_3/ReshapeReshape4sequential_3/rot_inv_pool_2/PartitionedCall:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:�����������
*sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
sequential_3/dense_6/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
+sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_3/dense_6/BiasAddBiasAdd%sequential_3/dense_6/MatMul:product:03sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� z
sequential_3/dense_6/ReluRelu%sequential_3/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
*sequential_3/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
sequential_3/dense_7/MatMulMatMul'sequential_3/dense_6/Relu:activations:02sequential_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+sequential_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/dense_7/BiasAddBiasAdd%sequential_3/dense_7/MatMul:product:03sequential_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%sequential_3/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^sequential_3/dense_6/BiasAdd/ReadVariableOp+^sequential_3/dense_6/MatMul/ReadVariableOp,^sequential_3/dense_7/BiasAdd/ReadVariableOp+^sequential_3/dense_7/MatMul/ReadVariableOp9^sequential_3/rot_equiv_conv2d_10/StatefulPartitionedCall9^sequential_3/rot_equiv_conv2d_11/StatefulPartitionedCall9^sequential_3/rot_equiv_conv2d_12/StatefulPartitionedCall9^sequential_3/rot_equiv_conv2d_13/StatefulPartitionedCall9^sequential_3/rot_equiv_conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2Z
+sequential_3/dense_6/BiasAdd/ReadVariableOp+sequential_3/dense_6/BiasAdd/ReadVariableOp2X
*sequential_3/dense_6/MatMul/ReadVariableOp*sequential_3/dense_6/MatMul/ReadVariableOp2Z
+sequential_3/dense_7/BiasAdd/ReadVariableOp+sequential_3/dense_7/BiasAdd/ReadVariableOp2X
*sequential_3/dense_7/MatMul/ReadVariableOp*sequential_3/dense_7/MatMul/ReadVariableOp2t
8sequential_3/rot_equiv_conv2d_10/StatefulPartitionedCall8sequential_3/rot_equiv_conv2d_10/StatefulPartitionedCall2t
8sequential_3/rot_equiv_conv2d_11/StatefulPartitionedCall8sequential_3/rot_equiv_conv2d_11/StatefulPartitionedCall2t
8sequential_3/rot_equiv_conv2d_12/StatefulPartitionedCall8sequential_3/rot_equiv_conv2d_12/StatefulPartitionedCall2t
8sequential_3/rot_equiv_conv2d_13/StatefulPartitionedCall8sequential_3/rot_equiv_conv2d_13/StatefulPartitionedCall2t
8sequential_3/rot_equiv_conv2d_14/StatefulPartitionedCall8sequential_3/rot_equiv_conv2d_14/StatefulPartitionedCall:l h
1
_output_shapes
:�����������
3
_user_specified_namerot_equiv_conv2d_10_input
�
f
J__inference_rot_inv_pool_2_layer_call_and_return_conditional_losses_494314

inputs
identity`
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:����������]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�>
�	
H__inference_sequential_3_layer_call_and_return_conditional_losses_631323

inputs4
rot_equiv_conv2d_10_631277: (
rot_equiv_conv2d_10_631279: 4
rot_equiv_conv2d_11_631283:  (
rot_equiv_conv2d_11_631285: 4
rot_equiv_conv2d_12_631289: @(
rot_equiv_conv2d_12_631291:@4
rot_equiv_conv2d_13_631295:@@(
rot_equiv_conv2d_13_631297:@5
rot_equiv_conv2d_14_631301:@�)
rot_equiv_conv2d_14_631303:	�9
&dense_6_matmul_readvariableop_resource:	� 5
'dense_6_biasadd_readvariableop_resource: 8
&dense_7_matmul_readvariableop_resource: 5
'dense_7_biasadd_readvariableop_resource:
identity��dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�+rot_equiv_conv2d_10/StatefulPartitionedCall�+rot_equiv_conv2d_11/StatefulPartitionedCall�+rot_equiv_conv2d_12/StatefulPartitionedCall�+rot_equiv_conv2d_13/StatefulPartitionedCall�+rot_equiv_conv2d_14/StatefulPartitionedCall�
+rot_equiv_conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsrot_equiv_conv2d_10_631277rot_equiv_conv2d_10_631279*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630646�
"rot_equiv_pool2d_8/PartitionedCallPartitionedCall4rot_equiv_conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630656�
+rot_equiv_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_8/PartitionedCall:output:0rot_equiv_conv2d_11_631283rot_equiv_conv2d_11_631285*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630666�
"rot_equiv_pool2d_9/PartitionedCallPartitionedCall4rot_equiv_conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630676�
+rot_equiv_conv2d_12/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_9/PartitionedCall:output:0rot_equiv_conv2d_12_631289rot_equiv_conv2d_12_631291*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630686�
#rot_equiv_pool2d_10/PartitionedCallPartitionedCall4rot_equiv_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630696�
+rot_equiv_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_10/PartitionedCall:output:0rot_equiv_conv2d_13_631295rot_equiv_conv2d_13_631297*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630706�
#rot_equiv_pool2d_11/PartitionedCallPartitionedCall4rot_equiv_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630716�
+rot_equiv_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_11/PartitionedCall:output:0rot_equiv_conv2d_14_631301rot_equiv_conv2d_14_631303*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630726�
rot_inv_pool_2/PartitionedCallPartitionedCall4rot_equiv_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630736`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten_3/ReshapeReshape'rot_inv_pool_2/PartitionedCall:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:�����������
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp,^rot_equiv_conv2d_10/StatefulPartitionedCall,^rot_equiv_conv2d_11/StatefulPartitionedCall,^rot_equiv_conv2d_12/StatefulPartitionedCall,^rot_equiv_conv2d_13/StatefulPartitionedCall,^rot_equiv_conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2Z
+rot_equiv_conv2d_10/StatefulPartitionedCall+rot_equiv_conv2d_10/StatefulPartitionedCall2Z
+rot_equiv_conv2d_11/StatefulPartitionedCall+rot_equiv_conv2d_11/StatefulPartitionedCall2Z
+rot_equiv_conv2d_12/StatefulPartitionedCall+rot_equiv_conv2d_12/StatefulPartitionedCall2Z
+rot_equiv_conv2d_13/StatefulPartitionedCall+rot_equiv_conv2d_13/StatefulPartitionedCall2Z
+rot_equiv_conv2d_14/StatefulPartitionedCall+rot_equiv_conv2d_14/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
O
3__inference_rot_equiv_pool2d_8_layer_call_fn_492961

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_rot_equiv_pool2d_8_layer_call_and_return_conditional_losses_492956l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������GG "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������� :] Y
5
_output_shapes#
!:����������� 
 
_user_specified_nameinputs
�
f
J__inference_rot_inv_pool_2_layer_call_and_return_conditional_losses_494849

inputs
identity`
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:����������]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�	
�
C__inference_dense_7_layer_call_and_return_conditional_losses_630826

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�9
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_630833

inputs4
rot_equiv_conv2d_10_630761: (
rot_equiv_conv2d_10_630763: 4
rot_equiv_conv2d_11_630767:  (
rot_equiv_conv2d_11_630769: 4
rot_equiv_conv2d_12_630773: @(
rot_equiv_conv2d_12_630775:@4
rot_equiv_conv2d_13_630779:@@(
rot_equiv_conv2d_13_630781:@5
rot_equiv_conv2d_14_630785:@�)
rot_equiv_conv2d_14_630787:	�!
dense_6_630811:	� 
dense_6_630813:  
dense_7_630827: 
dense_7_630829:
identity��dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�+rot_equiv_conv2d_10/StatefulPartitionedCall�+rot_equiv_conv2d_11/StatefulPartitionedCall�+rot_equiv_conv2d_12/StatefulPartitionedCall�+rot_equiv_conv2d_13/StatefulPartitionedCall�+rot_equiv_conv2d_14/StatefulPartitionedCall�
+rot_equiv_conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsrot_equiv_conv2d_10_630761rot_equiv_conv2d_10_630763*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630646�
"rot_equiv_pool2d_8/PartitionedCallPartitionedCall4rot_equiv_conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630656�
+rot_equiv_conv2d_11/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_8/PartitionedCall:output:0rot_equiv_conv2d_11_630767rot_equiv_conv2d_11_630769*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630666�
"rot_equiv_pool2d_9/PartitionedCallPartitionedCall4rot_equiv_conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������"" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630676�
+rot_equiv_conv2d_12/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_9/PartitionedCall:output:0rot_equiv_conv2d_12_630773rot_equiv_conv2d_12_630775*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630686�
#rot_equiv_pool2d_10/PartitionedCallPartitionedCall4rot_equiv_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630696�
+rot_equiv_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_10/PartitionedCall:output:0rot_equiv_conv2d_13_630779rot_equiv_conv2d_13_630781*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630706�
#rot_equiv_pool2d_11/PartitionedCallPartitionedCall4rot_equiv_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630716�
+rot_equiv_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall,rot_equiv_pool2d_11/PartitionedCall:output:0rot_equiv_conv2d_14_630785rot_equiv_conv2d_14_630787*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630726�
rot_inv_pool_2/PartitionedCallPartitionedCall4rot_equiv_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *2
f-R+
)__inference_restored_function_body_630736�
flatten_3/PartitionedCallPartitionedCall'rot_inv_pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_630797�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_630811dense_6_630813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_630810�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_630827dense_7_630829*
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
GPU2*0J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_630826w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^rot_equiv_conv2d_10/StatefulPartitionedCall,^rot_equiv_conv2d_11/StatefulPartitionedCall,^rot_equiv_conv2d_12/StatefulPartitionedCall,^rot_equiv_conv2d_13/StatefulPartitionedCall,^rot_equiv_conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+rot_equiv_conv2d_10/StatefulPartitionedCall+rot_equiv_conv2d_10/StatefulPartitionedCall2Z
+rot_equiv_conv2d_11/StatefulPartitionedCall+rot_equiv_conv2d_11/StatefulPartitionedCall2Z
+rot_equiv_conv2d_12/StatefulPartitionedCall+rot_equiv_conv2d_12/StatefulPartitionedCall2Z
+rot_equiv_conv2d_13/StatefulPartitionedCall+rot_equiv_conv2d_13/StatefulPartitionedCall2Z
+rot_equiv_conv2d_14/StatefulPartitionedCall+rot_equiv_conv2d_14/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�6
j
N__inference_rot_equiv_pool2d_9_layer_call_and_return_conditional_losses_491736

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������EE �
max_pooling2d_13/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:���������"" *
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������EE �
max_pooling2d_13/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:���������"" *
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������EE �
max_pooling2d_13/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:���������"" *
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������EE �
max_pooling2d_13/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:���������"" *
ksize
*
paddingVALID*
strides
�
stackPack!max_pooling2d_13/MaxPool:output:0#max_pooling2d_13/MaxPool_1:output:0#max_pooling2d_13/MaxPool_2:output:0#max_pooling2d_13/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:���������"" *
axis���������b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:���������"" "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������EE :[ W
3
_output_shapes!
:���������EE 
 
_user_specified_nameinputs
�
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_631382

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_sequential_3_layer_call_fn_631034
rot_equiv_conv2d_10_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�
	unknown_9:	� 

unknown_10: 

unknown_11: 

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_630970o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
1
_output_shapes
:�����������
3
_user_specified_namerot_equiv_conv2d_10_input
�
�
4__inference_rot_equiv_conv2d_11_layer_call_fn_491641

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_conv2d_11_layer_call_and_return_conditional_losses_491634`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 {
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������EE "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������GG : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������GG 
 
_user_specified_nameinputs
�	
�
C__inference_dense_7_layer_call_and_return_conditional_losses_631421

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�H
�
O__inference_rot_equiv_conv2d_10_layer_call_and_return_conditional_losses_491815

inputs=
#convolution_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
convolutionConv2Dinputs"convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides

Rank/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:t
TensorScatterUpdate/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       l
TensorScatterUpdate/updatesConst*
_output_shapes
:*
dtype0*
valueB"       �
TensorScatterUpdateTensorScatterUpdaterange:output:0$TensorScatterUpdate/indices:output:0$TensorScatterUpdate/updates:output:0*
T0*
Tindices0*
_output_shapes
:z
ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:x
	ReverseV2	ReverseV2ReadVariableOp:value:0ReverseV2/axis:output:0*
T0*&
_output_shapes
: y
	transpose	TransposeReverseV2:output:0TensorScatterUpdate:output:0*
T0*&
_output_shapes
: �
convolution_1Conv2Dinputstranspose:y:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :m
range_1Rangerange_1/start:output:0Rank_2:output:0range_1/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_1/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      n
TensorScatterUpdate_1/updatesConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorScatterUpdate_1TensorScatterUpdaterange_1:output:0&TensorScatterUpdate_1/indices:output:0&TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:�
transpose_1	Transposeconvolution_1:output:0TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:����������� H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:�
ReverseV2_1	ReverseV2transpose_1:y:0ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:����������� �
Rank_4/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_4Const*
_output_shapes
: *
dtype0*
value	B :|
ReadVariableOp_1ReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0H
Rank_5Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_2/axisConst*
_output_shapes
:*
dtype0*
valueB: ~
ReverseV2_2	ReverseV2ReadVariableOp_1:value:0ReverseV2_2/axis:output:0*
T0*&
_output_shapes
: H
Rank_6Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_3/axisConst*
_output_shapes
:*
dtype0*
valueB:z
ReverseV2_3	ReverseV2ReverseV2_2:output:0ReverseV2_3/axis:output:0*
T0*&
_output_shapes
: �
convolution_2Conv2DinputsReverseV2_3:output:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
H
Rank_7Const*
_output_shapes
: *
dtype0*
value	B :H
Rank_8Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_4/axisConst*
_output_shapes
:*
dtype0*
valueB:�
ReverseV2_4	ReverseV2convolution_2:output:0ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:����������� H
Rank_9Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:�
ReverseV2_5	ReverseV2ReverseV2_4:output:0ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:����������� �
Rank_10/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0I
Rank_10Const*
_output_shapes
: *
dtype0*
value	B :O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
range_2Rangerange_2/start:output:0Rank_10:output:0range_2/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_2/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       n
TensorScatterUpdate_2/updatesConst*
_output_shapes
:*
dtype0*
valueB"       �
TensorScatterUpdate_2TensorScatterUpdaterange_2:output:0&TensorScatterUpdate_2/indices:output:0&TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:|
ReadVariableOp_2ReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
transpose_2	TransposeReadVariableOp_2:value:0TensorScatterUpdate_2:output:0*
T0*&
_output_shapes
: I
Rank_11Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_6/axisConst*
_output_shapes
:*
dtype0*
valueB:u
ReverseV2_6	ReverseV2transpose_2:y:0ReverseV2_6/axis:output:0*
T0*&
_output_shapes
: �
convolution_3Conv2DinputsReverseV2_6:output:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
I
Rank_12Const*
_output_shapes
: *
dtype0*
value	B :O
range_3/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
range_3Rangerange_3/start:output:0Rank_12:output:0range_3/delta:output:0*
_output_shapes
:v
TensorScatterUpdate_3/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      n
TensorScatterUpdate_3/updatesConst*
_output_shapes
:*
dtype0*
valueB"      �
TensorScatterUpdate_3TensorScatterUpdaterange_3:output:0&TensorScatterUpdate_3/indices:output:0&TensorScatterUpdate_3/updates:output:0*
T0*
Tindices0*
_output_shapes
:I
Rank_13Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_7/axisConst*
_output_shapes
:*
dtype0*
valueB:�
ReverseV2_7	ReverseV2convolution_3:output:0ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:����������� �
transpose_3	TransposeReverseV2_7:output:0TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:����������� �
stackPackconvolution:output:0ReverseV2_1:output:0ReverseV2_5:output:0transpose_3:y:0*
N*
T0*5
_output_shapes#
!:����������� *
axis���������\
ReluRelustack:output:0*
T0*5
_output_shapes#
!:����������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:����������� �
NoOpNoOp^BiasAdd/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^convolution/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_228
convolution/ReadVariableOpconvolution/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
)__inference_restored_function_body_630726

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_conv2d_14_layer_call_and_return_conditional_losses_494450|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_631344

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_631356

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
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
4__inference_rot_equiv_conv2d_12_layer_call_fn_492126

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_conv2d_12_layer_call_and_return_conditional_losses_492119`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 {
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������"" : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������"" 
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_631461

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
P
4__inference_rot_equiv_pool2d_10_layer_call_fn_494599

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_pool2d_10_layer_call_and_return_conditional_losses_494594l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @:[ W
3
_output_shapes!
:���������  @
 
_user_specified_nameinputs
�
�
)__inference_restored_function_body_630686

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*3
_output_shapes!
:���������  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_rot_equiv_conv2d_12_layer_call_and_return_conditional_losses_492049{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������"" : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������"" 
 
_user_specified_nameinputs
�C
�
O__inference_rot_equiv_conv2d_14_layer_call_and_return_conditional_losses_494450

inputs>
#convolution_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�convolution_1/ReadVariableOp�convolution_2/ReadVariableOp�convolution_3/ReadVariableOpG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:���������@�
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*4
_output_shapes"
 :����������*
axis���������[
ReluRelustack:output:0*
T0*4
_output_shapes"
 :����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�6
j
N__inference_rot_equiv_pool2d_8_layer_call_and_return_conditional_losses_495185

inputs
identityG
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R K
ShapeShapeinputs*
T0*
_output_shapes
:*
out_type0	f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RS
subSubstrided_slice:output:0sub/y:output:0*
T0	*
_output_shapes
: Z
clip_by_value/MinimumMinimumConst:output:0sub:z:0*
T0	*
_output_shapes
: Q
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R n
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
: X
GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:����������� �
max_pooling2d_12/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:���������GG *
ksize
*
paddingVALID*
strides
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_1Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_1/MinimumMinimumConst_1:output:0	sub_1:z:0*
T0	*
_output_shapes
: S
clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:����������� �
max_pooling2d_12/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:���������GG *
ksize
*
paddingVALID*
strides
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_2Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_2/MinimumMinimumConst_2:output:0	sub_2:z:0*
T0	*
_output_shapes
: S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:����������� �
max_pooling2d_12/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:���������GG *
ksize
*
paddingVALID*
strides
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RM
Shape_3Shapeinputs*
T0*
_output_shapes
:*
out_type0	h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_3:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: `
clip_by_value_3/MinimumMinimumConst_3:output:0	sub_3:z:0*
T0	*
_output_shapes
: S
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R t
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes
: Z
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:����������� �
max_pooling2d_12/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:���������GG *
ksize
*
paddingVALID*
strides
�
stackPack!max_pooling2d_12/MaxPool:output:0#max_pooling2d_12/MaxPool_1:output:0#max_pooling2d_12/MaxPool_2:output:0#max_pooling2d_12/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:���������GG *
axis���������b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:���������GG "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������� :] Y
5
_output_shapes#
!:����������� 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
i
rot_equiv_conv2d_10_inputL
+serving_default_rot_equiv_conv2d_10_input:0�����������;
dense_70
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
	filt_base
bias
# _self_saveable_object_factories"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'pool
#(_self_saveable_object_factories"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
/	filt_base
0bias
#1_self_saveable_object_factories"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8pool
#9_self_saveable_object_factories"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
@	filt_base
Abias
#B_self_saveable_object_factories"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Ipool
#J_self_saveable_object_factories"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Q	filt_base
Rbias
#S_self_saveable_object_factories"
_tf_keras_layer
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Zpool
#[_self_saveable_object_factories"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
b	filt_base
cbias
#d_self_saveable_object_factories"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
#k_self_saveable_object_factories"
_tf_keras_layer
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses
#r_self_saveable_object_factories"
_tf_keras_layer
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias
#{_self_saveable_object_factories"
_tf_keras_layer
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories"
_tf_keras_layer
�
0
1
/2
03
@4
A5
Q6
R7
b8
c9
y10
z11
�12
�13"
trackable_list_wrapper
�
0
1
/2
03
@4
A5
Q6
R7
b8
c9
y10
z11
�12
�13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
-__inference_sequential_3_layer_call_fn_630864
-__inference_sequential_3_layer_call_fn_631192
-__inference_sequential_3_layer_call_fn_631225
-__inference_sequential_3_layer_call_fn_631034�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
H__inference_sequential_3_layer_call_and_return_conditional_losses_631274
H__inference_sequential_3_layer_call_and_return_conditional_losses_631323
H__inference_sequential_3_layer_call_and_return_conditional_losses_631079
H__inference_sequential_3_layer_call_and_return_conditional_losses_631124�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
!__inference__wrapped_model_630754rot_equiv_conv2d_10_input"�
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
"
	optimizer
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_rot_equiv_conv2d_10_layer_call_fn_494838�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_rot_equiv_conv2d_10_layer_call_and_return_conditional_losses_491815�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
4:2 2rot_equiv_conv2d_10/kernel
&:$ 2rot_equiv_conv2d_10/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_rot_equiv_pool2d_8_layer_call_fn_492961�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_rot_equiv_pool2d_8_layer_call_and_return_conditional_losses_495185�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_rot_equiv_conv2d_11_layer_call_fn_491641�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_rot_equiv_conv2d_11_layer_call_and_return_conditional_losses_493720�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
4:2  2rot_equiv_conv2d_11/kernel
&:$ 2rot_equiv_conv2d_11/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_rot_equiv_pool2d_9_layer_call_fn_494380�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_rot_equiv_pool2d_9_layer_call_and_return_conditional_losses_491736�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_rot_equiv_conv2d_12_layer_call_fn_492126�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_rot_equiv_conv2d_12_layer_call_and_return_conditional_losses_492049�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
4:2 @2rot_equiv_conv2d_12/kernel
&:$@2rot_equiv_conv2d_12/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_rot_equiv_pool2d_10_layer_call_fn_494599�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_rot_equiv_pool2d_10_layer_call_and_return_conditional_losses_492187�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_rot_equiv_conv2d_13_layer_call_fn_494676�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_rot_equiv_conv2d_13_layer_call_and_return_conditional_losses_495124�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
4:2@@2rot_equiv_conv2d_13/kernel
&:$@2rot_equiv_conv2d_13/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_rot_equiv_pool2d_11_layer_call_fn_494533�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_rot_equiv_pool2d_11_layer_call_and_return_conditional_losses_494752�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_rot_equiv_conv2d_14_layer_call_fn_493650�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_rot_equiv_conv2d_14_layer_call_and_return_conditional_losses_494450�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
5:3@�2rot_equiv_conv2d_14/kernel
':%�2rot_equiv_conv2d_14/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_rot_inv_pool_2_layer_call_fn_494854�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_rot_inv_pool_2_layer_call_and_return_conditional_losses_494314�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_flatten_3_layer_call_fn_631376�
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
 z�trace_0
�
�trace_02�
E__inference_flatten_3_layer_call_and_return_conditional_losses_631382�
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
 z�trace_0
 "
trackable_dict_wrapper
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_6_layer_call_fn_631391�
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
 z�trace_0
�
�trace_02�
C__inference_dense_6_layer_call_and_return_conditional_losses_631402�
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
 z�trace_0
!:	� 2dense_6/kernel
: 2dense_6/bias
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_7_layer_call_fn_631411�
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
 z�trace_0
�
�trace_02�
C__inference_dense_7_layer_call_and_return_conditional_losses_631421�
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
 z�trace_0
 : 2dense_7/kernel
:2dense_7/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_3_layer_call_fn_630864rot_equiv_conv2d_10_input"�
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
-__inference_sequential_3_layer_call_fn_631192inputs"�
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
-__inference_sequential_3_layer_call_fn_631225inputs"�
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
�B�
-__inference_sequential_3_layer_call_fn_631034rot_equiv_conv2d_10_input"�
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
�B�
H__inference_sequential_3_layer_call_and_return_conditional_losses_631274inputs"�
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
�B�
H__inference_sequential_3_layer_call_and_return_conditional_losses_631323inputs"�
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
�B�
H__inference_sequential_3_layer_call_and_return_conditional_losses_631079rot_equiv_conv2d_10_input"�
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
�B�
H__inference_sequential_3_layer_call_and_return_conditional_losses_631124rot_equiv_conv2d_10_input"�
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
$__inference_signature_wrapper_631159rot_equiv_conv2d_10_input"�
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
�B�
4__inference_rot_equiv_conv2d_10_layer_call_fn_494838inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_rot_equiv_conv2d_10_layer_call_and_return_conditional_losses_491815inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_rot_equiv_pool2d_8_layer_call_fn_492961inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_rot_equiv_pool2d_8_layer_call_and_return_conditional_losses_495185inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_12_layer_call_fn_631426�
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
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_631431�
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
 z�trace_0
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
�B�
4__inference_rot_equiv_conv2d_11_layer_call_fn_491641inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_rot_equiv_conv2d_11_layer_call_and_return_conditional_losses_493720inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_rot_equiv_pool2d_9_layer_call_fn_494380inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_rot_equiv_pool2d_9_layer_call_and_return_conditional_losses_491736inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_13_layer_call_fn_631436�
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
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_631441�
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
 z�trace_0
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
�B�
4__inference_rot_equiv_conv2d_12_layer_call_fn_492126inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_rot_equiv_conv2d_12_layer_call_and_return_conditional_losses_492049inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_rot_equiv_pool2d_10_layer_call_fn_494599inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_rot_equiv_pool2d_10_layer_call_and_return_conditional_losses_492187inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_14_layer_call_fn_631446�
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
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_631451�
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
 z�trace_0
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
�B�
4__inference_rot_equiv_conv2d_13_layer_call_fn_494676inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_rot_equiv_conv2d_13_layer_call_and_return_conditional_losses_495124inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
'
Z0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_rot_equiv_pool2d_11_layer_call_fn_494533inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_rot_equiv_pool2d_11_layer_call_and_return_conditional_losses_494752inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_15_layer_call_fn_631456�
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
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_631461�
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
 z�trace_0
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
�B�
4__inference_rot_equiv_conv2d_14_layer_call_fn_493650inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_rot_equiv_conv2d_14_layer_call_and_return_conditional_losses_494450inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
/__inference_rot_inv_pool_2_layer_call_fn_494854inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_rot_inv_pool_2_layer_call_and_return_conditional_losses_494314inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
*__inference_flatten_3_layer_call_fn_631376inputs"�
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_631382inputs"�
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
�B�
(__inference_dense_6_layer_call_fn_631391inputs"�
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
C__inference_dense_6_layer_call_and_return_conditional_losses_631402inputs"�
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
�B�
(__inference_dense_7_layer_call_fn_631411inputs"�
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
C__inference_dense_7_layer_call_and_return_conditional_losses_631421inputs"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
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
�B�
1__inference_max_pooling2d_12_layer_call_fn_631426inputs"�
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
�B�
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_631431inputs"�
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
�B�
1__inference_max_pooling2d_13_layer_call_fn_631436inputs"�
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
�B�
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_631441inputs"�
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
�B�
1__inference_max_pooling2d_14_layer_call_fn_631446inputs"�
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
�B�
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_631451inputs"�
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
�B�
1__inference_max_pooling2d_15_layer_call_fn_631456inputs"�
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
�B�
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_631461inputs"�
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
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
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
trackable_dict_wrapper�
!__inference__wrapped_model_630754�/0@AQRbcyz��L�I
B�?
=�:
rot_equiv_conv2d_10_input�����������
� "1�.
,
dense_7!�
dense_7����������
C__inference_dense_6_layer_call_and_return_conditional_losses_631402]yz0�-
&�#
!�
inputs����������
� "%�"
�
0��������� 
� |
(__inference_dense_6_layer_call_fn_631391Pyz0�-
&�#
!�
inputs����������
� "���������� �
C__inference_dense_7_layer_call_and_return_conditional_losses_631421^��/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� }
(__inference_dense_7_layer_call_fn_631411Q��/�,
%�"
 �
inputs��������� 
� "�����������
E__inference_flatten_3_layer_call_and_return_conditional_losses_631382b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
*__inference_flatten_3_layer_call_fn_631376U8�5
.�+
)�&
inputs����������
� "������������
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_631431�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_12_layer_call_fn_631426�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_631441�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_13_layer_call_fn_631436�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_631451�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_14_layer_call_fn_631446�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_631461�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_15_layer_call_fn_631456�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
O__inference_rot_equiv_conv2d_10_layer_call_and_return_conditional_losses_491815t9�6
/�,
*�'
inputs�����������
� "3�0
)�&
0����������� 
� �
4__inference_rot_equiv_conv2d_10_layer_call_fn_494838g9�6
/�,
*�'
inputs�����������
� "&�#����������� �
O__inference_rot_equiv_conv2d_11_layer_call_and_return_conditional_losses_493720t/0;�8
1�.
,�)
inputs���������GG 
� "1�.
'�$
0���������EE 
� �
4__inference_rot_equiv_conv2d_11_layer_call_fn_491641g/0;�8
1�.
,�)
inputs���������GG 
� "$�!���������EE �
O__inference_rot_equiv_conv2d_12_layer_call_and_return_conditional_losses_492049t@A;�8
1�.
,�)
inputs���������"" 
� "1�.
'�$
0���������  @
� �
4__inference_rot_equiv_conv2d_12_layer_call_fn_492126g@A;�8
1�.
,�)
inputs���������"" 
� "$�!���������  @�
O__inference_rot_equiv_conv2d_13_layer_call_and_return_conditional_losses_495124tQR;�8
1�.
,�)
inputs���������@
� "1�.
'�$
0���������@
� �
4__inference_rot_equiv_conv2d_13_layer_call_fn_494676gQR;�8
1�.
,�)
inputs���������@
� "$�!���������@�
O__inference_rot_equiv_conv2d_14_layer_call_and_return_conditional_losses_494450ubc;�8
1�.
,�)
inputs���������@
� "2�/
(�%
0����������
� �
4__inference_rot_equiv_conv2d_14_layer_call_fn_493650hbc;�8
1�.
,�)
inputs���������@
� "%�"�����������
O__inference_rot_equiv_pool2d_10_layer_call_and_return_conditional_losses_492187p;�8
1�.
,�)
inputs���������  @
� "1�.
'�$
0���������@
� �
4__inference_rot_equiv_pool2d_10_layer_call_fn_494599c;�8
1�.
,�)
inputs���������  @
� "$�!���������@�
O__inference_rot_equiv_pool2d_11_layer_call_and_return_conditional_losses_494752p;�8
1�.
,�)
inputs���������@
� "1�.
'�$
0���������@
� �
4__inference_rot_equiv_pool2d_11_layer_call_fn_494533c;�8
1�.
,�)
inputs���������@
� "$�!���������@�
N__inference_rot_equiv_pool2d_8_layer_call_and_return_conditional_losses_495185r=�:
3�0
.�+
inputs����������� 
� "1�.
'�$
0���������GG 
� �
3__inference_rot_equiv_pool2d_8_layer_call_fn_492961e=�:
3�0
.�+
inputs����������� 
� "$�!���������GG �
N__inference_rot_equiv_pool2d_9_layer_call_and_return_conditional_losses_491736p;�8
1�.
,�)
inputs���������EE 
� "1�.
'�$
0���������"" 
� �
3__inference_rot_equiv_pool2d_9_layer_call_fn_494380c;�8
1�.
,�)
inputs���������EE 
� "$�!���������"" �
J__inference_rot_inv_pool_2_layer_call_and_return_conditional_losses_494314n<�9
2�/
-�*
inputs����������
� ".�+
$�!
0����������
� �
/__inference_rot_inv_pool_2_layer_call_fn_494854a<�9
2�/
-�*
inputs����������
� "!������������
H__inference_sequential_3_layer_call_and_return_conditional_losses_631079�/0@AQRbcyz��T�Q
J�G
=�:
rot_equiv_conv2d_10_input�����������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_3_layer_call_and_return_conditional_losses_631124�/0@AQRbcyz��T�Q
J�G
=�:
rot_equiv_conv2d_10_input�����������
p

 
� "%�"
�
0���������
� �
H__inference_sequential_3_layer_call_and_return_conditional_losses_631274|/0@AQRbcyz��A�>
7�4
*�'
inputs�����������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_3_layer_call_and_return_conditional_losses_631323|/0@AQRbcyz��A�>
7�4
*�'
inputs�����������
p

 
� "%�"
�
0���������
� �
-__inference_sequential_3_layer_call_fn_630864�/0@AQRbcyz��T�Q
J�G
=�:
rot_equiv_conv2d_10_input�����������
p 

 
� "�����������
-__inference_sequential_3_layer_call_fn_631034�/0@AQRbcyz��T�Q
J�G
=�:
rot_equiv_conv2d_10_input�����������
p

 
� "�����������
-__inference_sequential_3_layer_call_fn_631192o/0@AQRbcyz��A�>
7�4
*�'
inputs�����������
p 

 
� "�����������
-__inference_sequential_3_layer_call_fn_631225o/0@AQRbcyz��A�>
7�4
*�'
inputs�����������
p

 
� "�����������
$__inference_signature_wrapper_631159�/0@AQRbcyz��i�f
� 
_�\
Z
rot_equiv_conv2d_10_input=�:
rot_equiv_conv2d_10_input�����������"1�.
,
dense_7!�
dense_7���������