║Ё'
╝Љ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
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
Џ
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
«
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
ї
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
ѓ
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
delete_old_dirsbool(ѕ
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
dtypetypeѕ
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
list(type)(0ѕ
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
list(type)(0ѕ
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
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.9.12v2.9.0-18-gd8ce9f9c3018╝Э#
ђ
Nadam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_1/bias/v
y
(Nadam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/v*
_output_shapes
:*
dtype0
ѕ
Nadam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameNadam/dense_1/kernel/v
Ђ
*Nadam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/v*
_output_shapes

: *
dtype0
|
Nadam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameNadam/dense/bias/v
u
&Nadam/dense/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense/bias/v*
_output_shapes
: *
dtype0
Ё
Nadam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ *%
shared_nameNadam/dense/kernel/v
~
(Nadam/dense/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/v*
_output_shapes
:	ђ *
dtype0
Ќ
Nadam/rot_equiv_conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*0
shared_name!Nadam/rot_equiv_conv2d_4/bias/v
љ
3Nadam/rot_equiv_conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_4/bias/v*
_output_shapes	
:ђ*
dtype0
Д
!Nadam/rot_equiv_conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*2
shared_name#!Nadam/rot_equiv_conv2d_4/kernel/v
а
5Nadam/rot_equiv_conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_4/kernel/v*'
_output_shapes
:@ђ*
dtype0
ќ
Nadam/rot_equiv_conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Nadam/rot_equiv_conv2d_3/bias/v
Ј
3Nadam/rot_equiv_conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_3/bias/v*
_output_shapes
:@*
dtype0
д
!Nadam/rot_equiv_conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!Nadam/rot_equiv_conv2d_3/kernel/v
Ъ
5Nadam/rot_equiv_conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0
ќ
Nadam/rot_equiv_conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Nadam/rot_equiv_conv2d_2/bias/v
Ј
3Nadam/rot_equiv_conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_2/bias/v*
_output_shapes
:@*
dtype0
д
!Nadam/rot_equiv_conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Nadam/rot_equiv_conv2d_2/kernel/v
Ъ
5Nadam/rot_equiv_conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
ќ
Nadam/rot_equiv_conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Nadam/rot_equiv_conv2d_1/bias/v
Ј
3Nadam/rot_equiv_conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_1/bias/v*
_output_shapes
: *
dtype0
д
!Nadam/rot_equiv_conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *2
shared_name#!Nadam/rot_equiv_conv2d_1/kernel/v
Ъ
5Nadam/rot_equiv_conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0
њ
Nadam/rot_equiv_conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameNadam/rot_equiv_conv2d/bias/v
І
1Nadam/rot_equiv_conv2d/bias/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d/bias/v*
_output_shapes
: *
dtype0
б
Nadam/rot_equiv_conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Nadam/rot_equiv_conv2d/kernel/v
Џ
3Nadam/rot_equiv_conv2d/kernel/v/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d/kernel/v*&
_output_shapes
: *
dtype0
ђ
Nadam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/dense_1/bias/m
y
(Nadam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/m*
_output_shapes
:*
dtype0
ѕ
Nadam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameNadam/dense_1/kernel/m
Ђ
*Nadam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/m*
_output_shapes

: *
dtype0
|
Nadam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameNadam/dense/bias/m
u
&Nadam/dense/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense/bias/m*
_output_shapes
: *
dtype0
Ё
Nadam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ *%
shared_nameNadam/dense/kernel/m
~
(Nadam/dense/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/m*
_output_shapes
:	ђ *
dtype0
Ќ
Nadam/rot_equiv_conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*0
shared_name!Nadam/rot_equiv_conv2d_4/bias/m
љ
3Nadam/rot_equiv_conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_4/bias/m*
_output_shapes	
:ђ*
dtype0
Д
!Nadam/rot_equiv_conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*2
shared_name#!Nadam/rot_equiv_conv2d_4/kernel/m
а
5Nadam/rot_equiv_conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_4/kernel/m*'
_output_shapes
:@ђ*
dtype0
ќ
Nadam/rot_equiv_conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Nadam/rot_equiv_conv2d_3/bias/m
Ј
3Nadam/rot_equiv_conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_3/bias/m*
_output_shapes
:@*
dtype0
д
!Nadam/rot_equiv_conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!Nadam/rot_equiv_conv2d_3/kernel/m
Ъ
5Nadam/rot_equiv_conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0
ќ
Nadam/rot_equiv_conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Nadam/rot_equiv_conv2d_2/bias/m
Ј
3Nadam/rot_equiv_conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_2/bias/m*
_output_shapes
:@*
dtype0
д
!Nadam/rot_equiv_conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Nadam/rot_equiv_conv2d_2/kernel/m
Ъ
5Nadam/rot_equiv_conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
ќ
Nadam/rot_equiv_conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Nadam/rot_equiv_conv2d_1/bias/m
Ј
3Nadam/rot_equiv_conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d_1/bias/m*
_output_shapes
: *
dtype0
д
!Nadam/rot_equiv_conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *2
shared_name#!Nadam/rot_equiv_conv2d_1/kernel/m
Ъ
5Nadam/rot_equiv_conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/rot_equiv_conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0
њ
Nadam/rot_equiv_conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameNadam/rot_equiv_conv2d/bias/m
І
1Nadam/rot_equiv_conv2d/bias/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d/bias/m*
_output_shapes
: *
dtype0
б
Nadam/rot_equiv_conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Nadam/rot_equiv_conv2d/kernel/m
Џ
3Nadam/rot_equiv_conv2d/kernel/m/Read/ReadVariableOpReadVariableOpNadam/rot_equiv_conv2d/kernel/m*&
_output_shapes
: *
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
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ *
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	ђ *
dtype0
Є
rot_equiv_conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*(
shared_namerot_equiv_conv2d_4/bias
ђ
+rot_equiv_conv2d_4/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_4/bias*
_output_shapes	
:ђ*
dtype0
Ќ
rot_equiv_conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ**
shared_namerot_equiv_conv2d_4/kernel
љ
-rot_equiv_conv2d_4/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_4/kernel*'
_output_shapes
:@ђ*
dtype0
є
rot_equiv_conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namerot_equiv_conv2d_3/bias

+rot_equiv_conv2d_3/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_3/bias*
_output_shapes
:@*
dtype0
ќ
rot_equiv_conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@**
shared_namerot_equiv_conv2d_3/kernel
Ј
-rot_equiv_conv2d_3/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_3/kernel*&
_output_shapes
:@@*
dtype0
є
rot_equiv_conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namerot_equiv_conv2d_2/bias

+rot_equiv_conv2d_2/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_2/bias*
_output_shapes
:@*
dtype0
ќ
rot_equiv_conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_namerot_equiv_conv2d_2/kernel
Ј
-rot_equiv_conv2d_2/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_2/kernel*&
_output_shapes
: @*
dtype0
є
rot_equiv_conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namerot_equiv_conv2d_1/bias

+rot_equiv_conv2d_1/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_1/bias*
_output_shapes
: *
dtype0
ќ
rot_equiv_conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_namerot_equiv_conv2d_1/kernel
Ј
-rot_equiv_conv2d_1/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d_1/kernel*&
_output_shapes
:  *
dtype0
ѓ
rot_equiv_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namerot_equiv_conv2d/bias
{
)rot_equiv_conv2d/bias/Read/ReadVariableOpReadVariableOprot_equiv_conv2d/bias*
_output_shapes
: *
dtype0
њ
rot_equiv_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namerot_equiv_conv2d/kernel
І
+rot_equiv_conv2d/kernel/Read/ReadVariableOpReadVariableOprot_equiv_conv2d/kernel*&
_output_shapes
: *
dtype0

NoOpNoOp
зЄ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ГЄ
valueбЄBъЄ BќЄ
«
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
signatures*
Е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	filt_base
bias*
ў
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%pool* 
Е
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,	filt_base
-bias*
ў
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4pool* 
Е
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;	filt_base
<bias*
ў
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cpool* 
Е
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
J	filt_base
Kbias*
ў
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Rpool* 
Е
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Y	filt_base
Zbias*
ј
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses* 
ј
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
д
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias*
д
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias*
j
0
1
,2
-3
;4
<5
J6
K7
Y8
Z9
m10
n11
u12
v13*
j
0
1
,2
-3
;4
<5
J6
K7
Y8
Z9
m10
n11
u12
v13*
* 
░
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
|trace_0
}trace_1
~trace_2
trace_3* 
:
ђtrace_0
Ђtrace_1
ѓtrace_2
Ѓtrace_3* 
* 
Ш
	ёiter
Ёbeta_1
єbeta_2

Єdecay
ѕlearning_rate
Ѕmomentum_cachemЦmд,mД-mе;mЕ<mфJmФKmгYmГZm«mm»nm░um▒vm▓v│v┤,vх-vХ;vи<vИJv╣Kv║Yv╗Zv╝mvйnvЙuv┐vv└*

іserving_default* 

0
1*

0
1*
* 
ў
Іnon_trainable_variables
їlayers
Їmetrics
 јlayer_regularization_losses
Јlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

љtrace_0* 

Љtrace_0* 
jd
VARIABLE_VALUErot_equiv_conv2d/kernel9layer_with_weights-0/filt_base/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUErot_equiv_conv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
њnon_trainable_variables
Њlayers
ћmetrics
 Ћlayer_regularization_losses
ќlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 

Ќtrace_0* 

ўtrace_0* 
ћ
Ў	variables
џtrainable_variables
Џregularization_losses
ю	keras_api
Ю__call__
+ъ&call_and_return_all_conditional_losses* 

,0
-1*

,0
-1*
* 
ў
Ъnon_trainable_variables
аlayers
Аmetrics
 бlayer_regularization_losses
Бlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

цtrace_0* 

Цtrace_0* 
lf
VARIABLE_VALUErot_equiv_conv2d_1/kernel9layer_with_weights-1/filt_base/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUErot_equiv_conv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
дnon_trainable_variables
Дlayers
еmetrics
 Еlayer_regularization_losses
фlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

Фtrace_0* 

гtrace_0* 
ћ
Г	variables
«trainable_variables
»regularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses* 

;0
<1*

;0
<1*
* 
ў
│non_trainable_variables
┤layers
хmetrics
 Хlayer_regularization_losses
иlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

Иtrace_0* 

╣trace_0* 
lf
VARIABLE_VALUErot_equiv_conv2d_2/kernel9layer_with_weights-2/filt_base/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUErot_equiv_conv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
║non_trainable_variables
╗layers
╝metrics
 йlayer_regularization_losses
Йlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

┐trace_0* 

└trace_0* 
ћ
┴	variables
┬trainable_variables
├regularization_losses
─	keras_api
┼__call__
+к&call_and_return_all_conditional_losses* 

J0
K1*

J0
K1*
* 
ў
Кnon_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

╠trace_0* 

═trace_0* 
lf
VARIABLE_VALUErot_equiv_conv2d_3/kernel9layer_with_weights-3/filt_base/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUErot_equiv_conv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
╬non_trainable_variables
¤layers
лmetrics
 Лlayer_regularization_losses
мlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses* 

Мtrace_0* 

нtrace_0* 
ћ
Н	variables
оtrainable_variables
Оregularization_losses
п	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses* 

Y0
Z1*

Y0
Z1*
* 
ў
█non_trainable_variables
▄layers
Пmetrics
 яlayer_regularization_losses
▀layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

Яtrace_0* 

рtrace_0* 
lf
VARIABLE_VALUErot_equiv_conv2d_4/kernel9layer_with_weights-4/filt_base/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUErot_equiv_conv2d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
Рnon_trainable_variables
сlayers
Сmetrics
 тlayer_regularization_losses
Тlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

уtrace_0* 

Уtrace_0* 
* 
* 
* 
ќ
жnon_trainable_variables
Жlayers
вmetrics
 Вlayer_regularization_losses
ьlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

Ьtrace_0* 

№trace_0* 

m0
n1*

m0
n1*
* 
ў
­non_trainable_variables
ыlayers
Ыmetrics
 зlayer_regularization_losses
Зlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

шtrace_0* 

Шtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

u0
v1*
* 
ў
эnon_trainable_variables
Эlayers
щmetrics
 Щlayer_regularization_losses
чlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

Чtrace_0* 

§trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
■0
 1*
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
MG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
%0* 
* 
* 
* 
* 
* 
* 
* 
* 
ю
ђnon_trainable_variables
Ђlayers
ѓmetrics
 Ѓlayer_regularization_losses
ёlayer_metrics
Ў	variables
џtrainable_variables
Џregularization_losses
Ю__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses* 

Ёtrace_0* 

єtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
	
40* 
* 
* 
* 
* 
* 
* 
* 
* 
ю
Єnon_trainable_variables
ѕlayers
Ѕmetrics
 іlayer_regularization_losses
Іlayer_metrics
Г	variables
«trainable_variables
»regularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses* 

їtrace_0* 

Їtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
	
C0* 
* 
* 
* 
* 
* 
* 
* 
* 
ю
јnon_trainable_variables
Јlayers
љmetrics
 Љlayer_regularization_losses
њlayer_metrics
┴	variables
┬trainable_variables
├regularization_losses
┼__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses* 

Њtrace_0* 

ћtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
	
R0* 
* 
* 
* 
* 
* 
* 
* 
* 
ю
Ћnon_trainable_variables
ќlayers
Ќmetrics
 ўlayer_regularization_losses
Ўlayer_metrics
Н	variables
оtrainable_variables
Оregularization_losses
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses* 

џtrace_0* 

Џtrace_0* 
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
ю	variables
Ю	keras_api

ъtotal

Ъcount*
M
а	variables
А	keras_api

бtotal

Бcount
ц
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
ъ0
Ъ1*

ю	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

б0
Б1*

а	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
Јѕ
VARIABLE_VALUENadam/rot_equiv_conv2d/kernel/mUlayer_with_weights-0/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѕЂ
VARIABLE_VALUENadam/rot_equiv_conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_1/kernel/mUlayer_with_weights-1/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUENadam/rot_equiv_conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_2/kernel/mUlayer_with_weights-2/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUENadam/rot_equiv_conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_3/kernel/mUlayer_with_weights-3/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUENadam/rot_equiv_conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_4/kernel/mUlayer_with_weights-4/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUENadam/rot_equiv_conv2d_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUENadam/dense/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/dense/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUENadam/dense_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Јѕ
VARIABLE_VALUENadam/rot_equiv_conv2d/kernel/vUlayer_with_weights-0/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѕЂ
VARIABLE_VALUENadam/rot_equiv_conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_1/kernel/vUlayer_with_weights-1/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUENadam/rot_equiv_conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_2/kernel/vUlayer_with_weights-2/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUENadam/rot_equiv_conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_3/kernel/vUlayer_with_weights-3/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUENadam/rot_equiv_conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE!Nadam/rot_equiv_conv2d_4/kernel/vUlayer_with_weights-4/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUENadam/rot_equiv_conv2d_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUENadam/dense/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/dense/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUENadam/dense_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUENadam/dense_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ю
&serving_default_rot_equiv_conv2d_inputPlaceholder*1
_output_shapes
:         љљ*
dtype0*&
shape:         љљ
ў
StatefulPartitionedCallStatefulPartitionedCall&serving_default_rot_equiv_conv2d_inputrot_equiv_conv2d/kernelrot_equiv_conv2d/biasrot_equiv_conv2d_1/kernelrot_equiv_conv2d_1/biasrot_equiv_conv2d_2/kernelrot_equiv_conv2d_2/biasrot_equiv_conv2d_3/kernelrot_equiv_conv2d_3/biasrot_equiv_conv2d_4/kernelrot_equiv_conv2d_4/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference_signature_wrapper_371016
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ж
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+rot_equiv_conv2d/kernel/Read/ReadVariableOp)rot_equiv_conv2d/bias/Read/ReadVariableOp-rot_equiv_conv2d_1/kernel/Read/ReadVariableOp+rot_equiv_conv2d_1/bias/Read/ReadVariableOp-rot_equiv_conv2d_2/kernel/Read/ReadVariableOp+rot_equiv_conv2d_2/bias/Read/ReadVariableOp-rot_equiv_conv2d_3/kernel/Read/ReadVariableOp+rot_equiv_conv2d_3/bias/Read/ReadVariableOp-rot_equiv_conv2d_4/kernel/Read/ReadVariableOp+rot_equiv_conv2d_4/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3Nadam/rot_equiv_conv2d/kernel/m/Read/ReadVariableOp1Nadam/rot_equiv_conv2d/bias/m/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_1/kernel/m/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_1/bias/m/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_2/kernel/m/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_2/bias/m/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_3/kernel/m/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_3/bias/m/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_4/kernel/m/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_4/bias/m/Read/ReadVariableOp(Nadam/dense/kernel/m/Read/ReadVariableOp&Nadam/dense/bias/m/Read/ReadVariableOp*Nadam/dense_1/kernel/m/Read/ReadVariableOp(Nadam/dense_1/bias/m/Read/ReadVariableOp3Nadam/rot_equiv_conv2d/kernel/v/Read/ReadVariableOp1Nadam/rot_equiv_conv2d/bias/v/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_1/kernel/v/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_1/bias/v/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_2/kernel/v/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_2/bias/v/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_3/kernel/v/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_3/bias/v/Read/ReadVariableOp5Nadam/rot_equiv_conv2d_4/kernel/v/Read/ReadVariableOp3Nadam/rot_equiv_conv2d_4/bias/v/Read/ReadVariableOp(Nadam/dense/kernel/v/Read/ReadVariableOp&Nadam/dense/bias/v/Read/ReadVariableOp*Nadam/dense_1/kernel/v/Read/ReadVariableOp(Nadam/dense_1/bias/v/Read/ReadVariableOpConst*A
Tin:
826	*
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
GPU2*0J 8ѓ *(
f#R!
__inference__traced_save_373206
Н
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerot_equiv_conv2d/kernelrot_equiv_conv2d/biasrot_equiv_conv2d_1/kernelrot_equiv_conv2d_1/biasrot_equiv_conv2d_2/kernelrot_equiv_conv2d_2/biasrot_equiv_conv2d_3/kernelrot_equiv_conv2d_3/biasrot_equiv_conv2d_4/kernelrot_equiv_conv2d_4/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotal_1count_1totalcountNadam/rot_equiv_conv2d/kernel/mNadam/rot_equiv_conv2d/bias/m!Nadam/rot_equiv_conv2d_1/kernel/mNadam/rot_equiv_conv2d_1/bias/m!Nadam/rot_equiv_conv2d_2/kernel/mNadam/rot_equiv_conv2d_2/bias/m!Nadam/rot_equiv_conv2d_3/kernel/mNadam/rot_equiv_conv2d_3/bias/m!Nadam/rot_equiv_conv2d_4/kernel/mNadam/rot_equiv_conv2d_4/bias/mNadam/dense/kernel/mNadam/dense/bias/mNadam/dense_1/kernel/mNadam/dense_1/bias/mNadam/rot_equiv_conv2d/kernel/vNadam/rot_equiv_conv2d/bias/v!Nadam/rot_equiv_conv2d_1/kernel/vNadam/rot_equiv_conv2d_1/bias/v!Nadam/rot_equiv_conv2d_2/kernel/vNadam/rot_equiv_conv2d_2/bias/v!Nadam/rot_equiv_conv2d_3/kernel/vNadam/rot_equiv_conv2d_3/bias/v!Nadam/rot_equiv_conv2d_4/kernel/vNadam/rot_equiv_conv2d_4/bias/vNadam/dense/kernel/vNadam/dense/bias/vNadam/dense_1/kernel/vNadam/dense_1/bias/v*@
Tin9
725*
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
GPU2*0J 8ѓ *+
f&R$
"__inference__traced_restore_373372Ю┘!
Ї
д
1__inference_rot_equiv_conv2d_layer_call_fn_372267

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:         јј *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_369992}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:         јј `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         љљ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         љљ
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_373017

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
С;
О
F__inference_sequential_layer_call_and_return_conditional_losses_370604

inputs1
rot_equiv_conv2d_369993: %
rot_equiv_conv2d_369995: 3
rot_equiv_conv2d_1_370132:  '
rot_equiv_conv2d_1_370134: 3
rot_equiv_conv2d_2_370271: @'
rot_equiv_conv2d_2_370273:@3
rot_equiv_conv2d_3_370410:@@'
rot_equiv_conv2d_3_370412:@4
rot_equiv_conv2d_4_370549:@ђ(
rot_equiv_conv2d_4_370551:	ђ
dense_370582:	ђ 
dense_370584:  
dense_1_370598: 
dense_1_370600:
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallб(rot_equiv_conv2d/StatefulPartitionedCallб*rot_equiv_conv2d_1/StatefulPartitionedCallб*rot_equiv_conv2d_2/StatefulPartitionedCallб*rot_equiv_conv2d_3/StatefulPartitionedCallб*rot_equiv_conv2d_4/StatefulPartitionedCallА
(rot_equiv_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsrot_equiv_conv2d_369993rot_equiv_conv2d_369995*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:         јј *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_369992ѓ
 rot_equiv_pool2d/PartitionedCallPartitionedCall1rot_equiv_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_370059╩
*rot_equiv_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)rot_equiv_pool2d/PartitionedCall:output:0rot_equiv_conv2d_1_370132rot_equiv_conv2d_1_370134*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_370131ѕ
"rot_equiv_pool2d_1/PartitionedCallPartitionedCall3rot_equiv_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         "" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_370198╠
*rot_equiv_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_1/PartitionedCall:output:0rot_equiv_conv2d_2_370271rot_equiv_conv2d_2_370273*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_370270ѕ
"rot_equiv_pool2d_2/PartitionedCallPartitionedCall3rot_equiv_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_370337╠
*rot_equiv_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_2/PartitionedCall:output:0rot_equiv_conv2d_3_370410rot_equiv_conv2d_3_370412*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_370409ѕ
"rot_equiv_pool2d_3/PartitionedCallPartitionedCall3rot_equiv_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_370476═
*rot_equiv_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_3/PartitionedCall:output:0rot_equiv_conv2d_4_370549rot_equiv_conv2d_4_370551*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_370548щ
rot_inv_pool/PartitionedCallPartitionedCall3rot_equiv_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_370560┘
flatten/PartitionedCallPartitionedCall%rot_inv_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_370568Ђ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_370582dense_370584*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_370581Ј
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_370598dense_1_370600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_370597w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         у
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall)^rot_equiv_conv2d/StatefulPartitionedCall+^rot_equiv_conv2d_1/StatefulPartitionedCall+^rot_equiv_conv2d_2/StatefulPartitionedCall+^rot_equiv_conv2d_3/StatefulPartitionedCall+^rot_equiv_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2T
(rot_equiv_conv2d/StatefulPartitionedCall(rot_equiv_conv2d/StatefulPartitionedCall2X
*rot_equiv_conv2d_1/StatefulPartitionedCall*rot_equiv_conv2d_1/StatefulPartitionedCall2X
*rot_equiv_conv2d_2/StatefulPartitionedCall*rot_equiv_conv2d_2/StatefulPartitionedCall2X
*rot_equiv_conv2d_3/StatefulPartitionedCall*rot_equiv_conv2d_3/StatefulPartitionedCall2X
*rot_equiv_conv2d_4/StatefulPartitionedCall*rot_equiv_conv2d_4/StatefulPartitionedCall:Y U
1
_output_shapes
:         љљ
 
_user_specified_nameinputs
к6
j
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_370198

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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE б
max_pooling2d_1/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:         "" *
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE д
max_pooling2d_1/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:         "" *
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE д
max_pooling2d_1/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:         "" *
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE д
max_pooling2d_1/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
э
stackPack max_pooling2d_1/MaxPool:output:0"max_pooling2d_1/MaxPool_1:output:0"max_pooling2d_1/MaxPool_2:output:0"max_pooling2d_1/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         "" *
axis■        b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:         "" "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         EE :[ W
3
_output_shapes!
:         EE 
 
_user_specified_nameinputs
ћ<
у
F__inference_sequential_layer_call_and_return_conditional_losses_370930
rot_equiv_conv2d_input1
rot_equiv_conv2d_370888: %
rot_equiv_conv2d_370890: 3
rot_equiv_conv2d_1_370894:  '
rot_equiv_conv2d_1_370896: 3
rot_equiv_conv2d_2_370900: @'
rot_equiv_conv2d_2_370902:@3
rot_equiv_conv2d_3_370906:@@'
rot_equiv_conv2d_3_370908:@4
rot_equiv_conv2d_4_370912:@ђ(
rot_equiv_conv2d_4_370914:	ђ
dense_370919:	ђ 
dense_370921:  
dense_1_370924: 
dense_1_370926:
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallб(rot_equiv_conv2d/StatefulPartitionedCallб*rot_equiv_conv2d_1/StatefulPartitionedCallб*rot_equiv_conv2d_2/StatefulPartitionedCallб*rot_equiv_conv2d_3/StatefulPartitionedCallб*rot_equiv_conv2d_4/StatefulPartitionedCall▒
(rot_equiv_conv2d/StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_inputrot_equiv_conv2d_370888rot_equiv_conv2d_370890*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:         јј *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_369992ѓ
 rot_equiv_pool2d/PartitionedCallPartitionedCall1rot_equiv_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_370059╩
*rot_equiv_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)rot_equiv_pool2d/PartitionedCall:output:0rot_equiv_conv2d_1_370894rot_equiv_conv2d_1_370896*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_370131ѕ
"rot_equiv_pool2d_1/PartitionedCallPartitionedCall3rot_equiv_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         "" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_370198╠
*rot_equiv_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_1/PartitionedCall:output:0rot_equiv_conv2d_2_370900rot_equiv_conv2d_2_370902*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_370270ѕ
"rot_equiv_pool2d_2/PartitionedCallPartitionedCall3rot_equiv_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_370337╠
*rot_equiv_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_2/PartitionedCall:output:0rot_equiv_conv2d_3_370906rot_equiv_conv2d_3_370908*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_370409ѕ
"rot_equiv_pool2d_3/PartitionedCallPartitionedCall3rot_equiv_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_370476═
*rot_equiv_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_3/PartitionedCall:output:0rot_equiv_conv2d_4_370912rot_equiv_conv2d_4_370914*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_370548щ
rot_inv_pool/PartitionedCallPartitionedCall3rot_equiv_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_370560┘
flatten/PartitionedCallPartitionedCall%rot_inv_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_370568Ђ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_370919dense_370921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_370581Ј
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_370924dense_1_370926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_370597w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         у
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall)^rot_equiv_conv2d/StatefulPartitionedCall+^rot_equiv_conv2d_1/StatefulPartitionedCall+^rot_equiv_conv2d_2/StatefulPartitionedCall+^rot_equiv_conv2d_3/StatefulPartitionedCall+^rot_equiv_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2T
(rot_equiv_conv2d/StatefulPartitionedCall(rot_equiv_conv2d/StatefulPartitionedCall2X
*rot_equiv_conv2d_1/StatefulPartitionedCall*rot_equiv_conv2d_1/StatefulPartitionedCall2X
*rot_equiv_conv2d_2/StatefulPartitionedCall*rot_equiv_conv2d_2/StatefulPartitionedCall2X
*rot_equiv_conv2d_3/StatefulPartitionedCall*rot_equiv_conv2d_3/StatefulPartitionedCall2X
*rot_equiv_conv2d_4/StatefulPartitionedCall*rot_equiv_conv2d_4/StatefulPartitionedCall:i e
1
_output_shapes
:         љљ
0
_user_specified_namerot_equiv_conv2d_input
│У
«
F__inference_sequential_layer_call_and_return_conditional_losses_372258

inputsN
4rot_equiv_conv2d_convolution_readvariableop_resource: >
0rot_equiv_conv2d_biasadd_readvariableop_resource: P
6rot_equiv_conv2d_1_convolution_readvariableop_resource:  @
2rot_equiv_conv2d_1_biasadd_readvariableop_resource: P
6rot_equiv_conv2d_2_convolution_readvariableop_resource: @@
2rot_equiv_conv2d_2_biasadd_readvariableop_resource:@P
6rot_equiv_conv2d_3_convolution_readvariableop_resource:@@@
2rot_equiv_conv2d_3_biasadd_readvariableop_resource:@Q
6rot_equiv_conv2d_4_convolution_readvariableop_resource:@ђA
2rot_equiv_conv2d_4_biasadd_readvariableop_resource:	ђ7
$dense_matmul_readvariableop_resource:	ђ 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpб'rot_equiv_conv2d/BiasAdd/ReadVariableOpбrot_equiv_conv2d/ReadVariableOpб!rot_equiv_conv2d/ReadVariableOp_1б!rot_equiv_conv2d/ReadVariableOp_2б+rot_equiv_conv2d/convolution/ReadVariableOpб)rot_equiv_conv2d_1/BiasAdd/ReadVariableOpб-rot_equiv_conv2d_1/convolution/ReadVariableOpб/rot_equiv_conv2d_1/convolution_1/ReadVariableOpб/rot_equiv_conv2d_1/convolution_2/ReadVariableOpб/rot_equiv_conv2d_1/convolution_3/ReadVariableOpб)rot_equiv_conv2d_2/BiasAdd/ReadVariableOpб-rot_equiv_conv2d_2/convolution/ReadVariableOpб/rot_equiv_conv2d_2/convolution_1/ReadVariableOpб/rot_equiv_conv2d_2/convolution_2/ReadVariableOpб/rot_equiv_conv2d_2/convolution_3/ReadVariableOpб)rot_equiv_conv2d_3/BiasAdd/ReadVariableOpб-rot_equiv_conv2d_3/convolution/ReadVariableOpб/rot_equiv_conv2d_3/convolution_1/ReadVariableOpб/rot_equiv_conv2d_3/convolution_2/ReadVariableOpб/rot_equiv_conv2d_3/convolution_3/ReadVariableOpб)rot_equiv_conv2d_4/BiasAdd/ReadVariableOpб-rot_equiv_conv2d_4/convolution/ReadVariableOpб/rot_equiv_conv2d_4/convolution_1/ReadVariableOpб/rot_equiv_conv2d_4/convolution_2/ReadVariableOpб/rot_equiv_conv2d_4/convolution_3/ReadVariableOpе
+rot_equiv_conv2d/convolution/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0╚
rot_equiv_conv2d/convolutionConv2Dinputs3rot_equiv_conv2d/convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
А
$rot_equiv_conv2d/Rank/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0W
rot_equiv_conv2d/RankConst*
_output_shapes
: *
dtype0*
value	B :^
rot_equiv_conv2d/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
rot_equiv_conv2d/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Е
rot_equiv_conv2d/rangeRange%rot_equiv_conv2d/range/start:output:0rot_equiv_conv2d/Rank:output:0%rot_equiv_conv2d/range/delta:output:0*
_output_shapes
:Ё
,rot_equiv_conv2d/TensorScatterUpdate/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       }
,rot_equiv_conv2d/TensorScatterUpdate/updatesConst*
_output_shapes
:*
dtype0*
valueB"        
$rot_equiv_conv2d/TensorScatterUpdateTensorScatterUpdaterot_equiv_conv2d/range:output:05rot_equiv_conv2d/TensorScatterUpdate/indices:output:05rot_equiv_conv2d/TensorScatterUpdate/updates:output:0*
T0*
Tindices0*
_output_shapes
:ю
rot_equiv_conv2d/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :i
rot_equiv_conv2d/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:Ф
rot_equiv_conv2d/ReverseV2	ReverseV2'rot_equiv_conv2d/ReadVariableOp:value:0(rot_equiv_conv2d/ReverseV2/axis:output:0*
T0*&
_output_shapes
: г
rot_equiv_conv2d/transpose	Transpose#rot_equiv_conv2d/ReverseV2:output:0-rot_equiv_conv2d/TensorScatterUpdate:output:0*
T0*&
_output_shapes
: х
rot_equiv_conv2d/convolution_1Conv2Dinputsrot_equiv_conv2d/transpose:y:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
Y
rot_equiv_conv2d/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :▒
rot_equiv_conv2d/range_1Range'rot_equiv_conv2d/range_1/start:output:0 rot_equiv_conv2d/Rank_2:output:0'rot_equiv_conv2d/range_1/delta:output:0*
_output_shapes
:Є
.rot_equiv_conv2d/TensorScatterUpdate_1/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      
.rot_equiv_conv2d/TensorScatterUpdate_1/updatesConst*
_output_shapes
:*
dtype0*
valueB"      Є
&rot_equiv_conv2d/TensorScatterUpdate_1TensorScatterUpdate!rot_equiv_conv2d/range_1:output:07rot_equiv_conv2d/TensorScatterUpdate_1/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:┐
rot_equiv_conv2d/transpose_1	Transpose'rot_equiv_conv2d/convolution_1:output:0/rot_equiv_conv2d/TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:         јј Y
rot_equiv_conv2d/Rank_3Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:│
rot_equiv_conv2d/ReverseV2_1	ReverseV2 rot_equiv_conv2d/transpose_1:y:0*rot_equiv_conv2d/ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:         јј Б
&rot_equiv_conv2d/Rank_4/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_4Const*
_output_shapes
: *
dtype0*
value	B :ъ
!rot_equiv_conv2d/ReadVariableOp_1ReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_5Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_2/axisConst*
_output_shapes
:*
dtype0*
valueB: ▒
rot_equiv_conv2d/ReverseV2_2	ReverseV2)rot_equiv_conv2d/ReadVariableOp_1:value:0*rot_equiv_conv2d/ReverseV2_2/axis:output:0*
T0*&
_output_shapes
: Y
rot_equiv_conv2d/Rank_6Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_3/axisConst*
_output_shapes
:*
dtype0*
valueB:Г
rot_equiv_conv2d/ReverseV2_3	ReverseV2%rot_equiv_conv2d/ReverseV2_2:output:0*rot_equiv_conv2d/ReverseV2_3/axis:output:0*
T0*&
_output_shapes
: ╝
rot_equiv_conv2d/convolution_2Conv2Dinputs%rot_equiv_conv2d/ReverseV2_3:output:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
Y
rot_equiv_conv2d/Rank_7Const*
_output_shapes
: *
dtype0*
value	B :Y
rot_equiv_conv2d/Rank_8Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_4/axisConst*
_output_shapes
:*
dtype0*
valueB:║
rot_equiv_conv2d/ReverseV2_4	ReverseV2'rot_equiv_conv2d/convolution_2:output:0*rot_equiv_conv2d/ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:         јј Y
rot_equiv_conv2d/Rank_9Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:И
rot_equiv_conv2d/ReverseV2_5	ReverseV2%rot_equiv_conv2d/ReverseV2_4:output:0*rot_equiv_conv2d/ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:         јј ц
'rot_equiv_conv2d/Rank_10/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Z
rot_equiv_conv2d/Rank_10Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :▓
rot_equiv_conv2d/range_2Range'rot_equiv_conv2d/range_2/start:output:0!rot_equiv_conv2d/Rank_10:output:0'rot_equiv_conv2d/range_2/delta:output:0*
_output_shapes
:Є
.rot_equiv_conv2d/TensorScatterUpdate_2/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       
.rot_equiv_conv2d/TensorScatterUpdate_2/updatesConst*
_output_shapes
:*
dtype0*
valueB"       Є
&rot_equiv_conv2d/TensorScatterUpdate_2TensorScatterUpdate!rot_equiv_conv2d/range_2:output:07rot_equiv_conv2d/TensorScatterUpdate_2/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:ъ
!rot_equiv_conv2d/ReadVariableOp_2ReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Х
rot_equiv_conv2d/transpose_2	Transpose)rot_equiv_conv2d/ReadVariableOp_2:value:0/rot_equiv_conv2d/TensorScatterUpdate_2:output:0*
T0*&
_output_shapes
: Z
rot_equiv_conv2d/Rank_11Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_6/axisConst*
_output_shapes
:*
dtype0*
valueB:е
rot_equiv_conv2d/ReverseV2_6	ReverseV2 rot_equiv_conv2d/transpose_2:y:0*rot_equiv_conv2d/ReverseV2_6/axis:output:0*
T0*&
_output_shapes
: ╝
rot_equiv_conv2d/convolution_3Conv2Dinputs%rot_equiv_conv2d/ReverseV2_6:output:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
Z
rot_equiv_conv2d/Rank_12Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :▓
rot_equiv_conv2d/range_3Range'rot_equiv_conv2d/range_3/start:output:0!rot_equiv_conv2d/Rank_12:output:0'rot_equiv_conv2d/range_3/delta:output:0*
_output_shapes
:Є
.rot_equiv_conv2d/TensorScatterUpdate_3/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      
.rot_equiv_conv2d/TensorScatterUpdate_3/updatesConst*
_output_shapes
:*
dtype0*
valueB"      Є
&rot_equiv_conv2d/TensorScatterUpdate_3TensorScatterUpdate!rot_equiv_conv2d/range_3:output:07rot_equiv_conv2d/TensorScatterUpdate_3/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_3/updates:output:0*
T0*
Tindices0*
_output_shapes
:Z
rot_equiv_conv2d/Rank_13Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_7/axisConst*
_output_shapes
:*
dtype0*
valueB:║
rot_equiv_conv2d/ReverseV2_7	ReverseV2'rot_equiv_conv2d/convolution_3:output:0*rot_equiv_conv2d/ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:         јј й
rot_equiv_conv2d/transpose_3	Transpose%rot_equiv_conv2d/ReverseV2_7:output:0/rot_equiv_conv2d/TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:         јј Њ
rot_equiv_conv2d/stackPack%rot_equiv_conv2d/convolution:output:0%rot_equiv_conv2d/ReverseV2_1:output:0%rot_equiv_conv2d/ReverseV2_5:output:0 rot_equiv_conv2d/transpose_3:y:0*
N*
T0*5
_output_shapes#
!:         јј *
axis■        ~
rot_equiv_conv2d/ReluRelurot_equiv_conv2d/stack:output:0*
T0*5
_output_shapes#
!:         јј ћ
'rot_equiv_conv2d/BiasAdd/ReadVariableOpReadVariableOp0rot_equiv_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╣
rot_equiv_conv2d/BiasAddBiasAdd#rot_equiv_conv2d/Relu:activations:0/rot_equiv_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:         јј X
rot_equiv_pool2d/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R w
rot_equiv_pool2d/ShapeShape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	w
$rot_equiv_pool2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        y
&rot_equiv_pool2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         p
&rot_equiv_pool2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
rot_equiv_pool2d/strided_sliceStridedSlicerot_equiv_pool2d/Shape:output:0-rot_equiv_pool2d/strided_slice/stack:output:0/rot_equiv_pool2d/strided_slice/stack_1:output:0/rot_equiv_pool2d/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskX
rot_equiv_pool2d/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rє
rot_equiv_pool2d/subSub'rot_equiv_pool2d/strided_slice:output:0rot_equiv_pool2d/sub/y:output:0*
T0	*
_output_shapes
: Ї
&rot_equiv_pool2d/clip_by_value/MinimumMinimumrot_equiv_pool2d/Const:output:0rot_equiv_pool2d/sub:z:0*
T0	*
_output_shapes
: b
 rot_equiv_pool2d/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R А
rot_equiv_pool2d/clip_by_valueMaximum*rot_equiv_pool2d/clip_by_value/Minimum:z:0)rot_equiv_pool2d/clip_by_value/y:output:0*
T0	*
_output_shapes
: i
rot_equiv_pool2d/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        З
rot_equiv_pool2d/GatherV2GatherV2!rot_equiv_conv2d/BiasAdd:output:0"rot_equiv_pool2d/clip_by_value:z:0'rot_equiv_pool2d/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ┬
&rot_equiv_pool2d/max_pooling2d/MaxPoolMaxPool"rot_equiv_pool2d/GatherV2:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_1Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d/strided_slice_1StridedSlice!rot_equiv_pool2d/Shape_1:output:0/rot_equiv_pool2d/strided_slice_1/stack:output:01rot_equiv_pool2d/strided_slice_1/stack_1:output:01rot_equiv_pool2d/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d/sub_1Sub)rot_equiv_pool2d/strided_slice_1:output:0!rot_equiv_pool2d/sub_1/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d/clip_by_value_1/MinimumMinimum!rot_equiv_pool2d/Const_1:output:0rot_equiv_pool2d/sub_1:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d/clip_by_value_1Maximum,rot_equiv_pool2d/clip_by_value_1/Minimum:z:0+rot_equiv_pool2d/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d/GatherV2_1GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_1:z:0)rot_equiv_pool2d/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј к
(rot_equiv_pool2d/max_pooling2d/MaxPool_1MaxPool$rot_equiv_pool2d/GatherV2_1:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_2Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d/strided_slice_2StridedSlice!rot_equiv_pool2d/Shape_2:output:0/rot_equiv_pool2d/strided_slice_2/stack:output:01rot_equiv_pool2d/strided_slice_2/stack_1:output:01rot_equiv_pool2d/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d/sub_2Sub)rot_equiv_pool2d/strided_slice_2:output:0!rot_equiv_pool2d/sub_2/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d/clip_by_value_2/MinimumMinimum!rot_equiv_pool2d/Const_2:output:0rot_equiv_pool2d/sub_2:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d/clip_by_value_2Maximum,rot_equiv_pool2d/clip_by_value_2/Minimum:z:0+rot_equiv_pool2d/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d/GatherV2_2GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_2:z:0)rot_equiv_pool2d/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј к
(rot_equiv_pool2d/max_pooling2d/MaxPool_2MaxPool$rot_equiv_pool2d/GatherV2_2:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_3Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d/strided_slice_3StridedSlice!rot_equiv_pool2d/Shape_3:output:0/rot_equiv_pool2d/strided_slice_3/stack:output:01rot_equiv_pool2d/strided_slice_3/stack_1:output:01rot_equiv_pool2d/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d/sub_3Sub)rot_equiv_pool2d/strided_slice_3:output:0!rot_equiv_pool2d/sub_3/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d/clip_by_value_3/MinimumMinimum!rot_equiv_pool2d/Const_3:output:0rot_equiv_pool2d/sub_3:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d/clip_by_value_3Maximum,rot_equiv_pool2d/clip_by_value_3/Minimum:z:0+rot_equiv_pool2d/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d/GatherV2_3GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_3:z:0)rot_equiv_pool2d/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј к
(rot_equiv_pool2d/max_pooling2d/MaxPool_3MaxPool$rot_equiv_pool2d/GatherV2_3:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
─
rot_equiv_pool2d/stackPack/rot_equiv_pool2d/max_pooling2d/MaxPool:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_1:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_2:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         GG *
axis■        Z
rot_equiv_conv2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R w
rot_equiv_conv2d_1/ShapeShaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_conv2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_conv2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_conv2d_1/strided_sliceStridedSlice!rot_equiv_conv2d_1/Shape:output:0/rot_equiv_conv2d_1/strided_slice/stack:output:01rot_equiv_conv2d_1/strided_slice/stack_1:output:01rot_equiv_conv2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_conv2d_1/subSub)rot_equiv_conv2d_1/strided_slice:output:0!rot_equiv_conv2d_1/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_conv2d_1/clip_by_value/MinimumMinimum!rot_equiv_conv2d_1/Const:output:0rot_equiv_conv2d_1/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_conv2d_1/clip_by_valueMaximum,rot_equiv_conv2d_1/clip_by_value/Minimum:z:0+rot_equiv_conv2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Ш
rot_equiv_conv2d_1/GatherV2GatherV2rot_equiv_pool2d/stack:output:0$rot_equiv_conv2d_1/clip_by_value:z:0)rot_equiv_conv2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG г
-rot_equiv_conv2d_1/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0У
rot_equiv_conv2d_1/convolutionConv2D$rot_equiv_conv2d_1/GatherV2:output:05rot_equiv_conv2d_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_1Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_1/strided_slice_1StridedSlice#rot_equiv_conv2d_1/Shape_1:output:01rot_equiv_conv2d_1/strided_slice_1/stack:output:03rot_equiv_conv2d_1/strided_slice_1/stack_1:output:03rot_equiv_conv2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_1/sub_1Sub+rot_equiv_conv2d_1/strided_slice_1:output:0#rot_equiv_conv2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_1/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_1/Const_1:output:0rot_equiv_conv2d_1/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_1/clip_by_value_1Maximum.rot_equiv_conv2d_1/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Ч
rot_equiv_conv2d_1/GatherV2_1GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_1:z:0+rot_equiv_conv2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG «
/rot_equiv_conv2d_1/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Ь
 rot_equiv_conv2d_1/convolution_1Conv2D&rot_equiv_conv2d_1/GatherV2_1:output:07rot_equiv_conv2d_1/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_2Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_1/strided_slice_2StridedSlice#rot_equiv_conv2d_1/Shape_2:output:01rot_equiv_conv2d_1/strided_slice_2/stack:output:03rot_equiv_conv2d_1/strided_slice_2/stack_1:output:03rot_equiv_conv2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_1/sub_2Sub+rot_equiv_conv2d_1/strided_slice_2:output:0#rot_equiv_conv2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_1/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_1/Const_2:output:0rot_equiv_conv2d_1/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_1/clip_by_value_2Maximum.rot_equiv_conv2d_1/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Ч
rot_equiv_conv2d_1/GatherV2_2GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_2:z:0+rot_equiv_conv2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG «
/rot_equiv_conv2d_1/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Ь
 rot_equiv_conv2d_1/convolution_2Conv2D&rot_equiv_conv2d_1/GatherV2_2:output:07rot_equiv_conv2d_1/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_3Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_1/strided_slice_3StridedSlice#rot_equiv_conv2d_1/Shape_3:output:01rot_equiv_conv2d_1/strided_slice_3/stack:output:03rot_equiv_conv2d_1/strided_slice_3/stack_1:output:03rot_equiv_conv2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_1/sub_3Sub+rot_equiv_conv2d_1/strided_slice_3:output:0#rot_equiv_conv2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_1/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_1/Const_3:output:0rot_equiv_conv2d_1/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_1/clip_by_value_3Maximum.rot_equiv_conv2d_1/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Ч
rot_equiv_conv2d_1/GatherV2_3GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_3:z:0+rot_equiv_conv2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG «
/rot_equiv_conv2d_1/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Ь
 rot_equiv_conv2d_1/convolution_3Conv2D&rot_equiv_conv2d_1/GatherV2_3:output:07rot_equiv_conv2d_1/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
д
rot_equiv_conv2d_1/stackPack'rot_equiv_conv2d_1/convolution:output:0)rot_equiv_conv2d_1/convolution_1:output:0)rot_equiv_conv2d_1/convolution_2:output:0)rot_equiv_conv2d_1/convolution_3:output:0*
N*
T0*3
_output_shapes!
:         EE *
axis■        ђ
rot_equiv_conv2d_1/ReluRelu!rot_equiv_conv2d_1/stack:output:0*
T0*3
_output_shapes!
:         EE ў
)rot_equiv_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
rot_equiv_conv2d_1/BiasAddBiasAdd%rot_equiv_conv2d_1/Relu:activations:01rot_equiv_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         EE Z
rot_equiv_pool2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_1/ShapeShape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d_1/strided_sliceStridedSlice!rot_equiv_pool2d_1/Shape:output:0/rot_equiv_pool2d_1/strided_slice/stack:output:01rot_equiv_pool2d_1/strided_slice/stack_1:output:01rot_equiv_pool2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d_1/subSub)rot_equiv_pool2d_1/strided_slice:output:0!rot_equiv_pool2d_1/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d_1/clip_by_value/MinimumMinimum!rot_equiv_pool2d_1/Const:output:0rot_equiv_pool2d_1/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d_1/clip_by_valueMaximum,rot_equiv_pool2d_1/clip_by_value/Minimum:z:0+rot_equiv_pool2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d_1/GatherV2GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0$rot_equiv_pool2d_1/clip_by_value:z:0)rot_equiv_pool2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE ╚
*rot_equiv_pool2d_1/max_pooling2d_1/MaxPoolMaxPool$rot_equiv_pool2d_1/GatherV2:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_1Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_1/strided_slice_1StridedSlice#rot_equiv_pool2d_1/Shape_1:output:01rot_equiv_pool2d_1/strided_slice_1/stack:output:03rot_equiv_pool2d_1/strided_slice_1/stack_1:output:03rot_equiv_pool2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_1/sub_1Sub+rot_equiv_pool2d_1/strided_slice_1:output:0#rot_equiv_pool2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_1/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_1/Const_1:output:0rot_equiv_pool2d_1/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_1/clip_by_value_1Maximum.rot_equiv_pool2d_1/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_1/GatherV2_1GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_1:z:0+rot_equiv_pool2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE ╠
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1MaxPool&rot_equiv_pool2d_1/GatherV2_1:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_2Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_1/strided_slice_2StridedSlice#rot_equiv_pool2d_1/Shape_2:output:01rot_equiv_pool2d_1/strided_slice_2/stack:output:03rot_equiv_pool2d_1/strided_slice_2/stack_1:output:03rot_equiv_pool2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_1/sub_2Sub+rot_equiv_pool2d_1/strided_slice_2:output:0#rot_equiv_pool2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_1/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_1/Const_2:output:0rot_equiv_pool2d_1/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_1/clip_by_value_2Maximum.rot_equiv_pool2d_1/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_1/GatherV2_2GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_2:z:0+rot_equiv_pool2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE ╠
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2MaxPool&rot_equiv_pool2d_1/GatherV2_2:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_3Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_1/strided_slice_3StridedSlice#rot_equiv_pool2d_1/Shape_3:output:01rot_equiv_pool2d_1/strided_slice_3/stack:output:03rot_equiv_pool2d_1/strided_slice_3/stack_1:output:03rot_equiv_pool2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_1/sub_3Sub+rot_equiv_pool2d_1/strided_slice_3:output:0#rot_equiv_pool2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_1/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_1/Const_3:output:0rot_equiv_pool2d_1/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_1/clip_by_value_3Maximum.rot_equiv_pool2d_1/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_1/GatherV2_3GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_3:z:0+rot_equiv_pool2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE ╠
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3MaxPool&rot_equiv_pool2d_1/GatherV2_3:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
о
rot_equiv_pool2d_1/stackPack3rot_equiv_pool2d_1/max_pooling2d_1/MaxPool:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         "" *
axis■        Z
rot_equiv_conv2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_2/ShapeShape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_conv2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_conv2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_conv2d_2/strided_sliceStridedSlice!rot_equiv_conv2d_2/Shape:output:0/rot_equiv_conv2d_2/strided_slice/stack:output:01rot_equiv_conv2d_2/strided_slice/stack_1:output:01rot_equiv_conv2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_conv2d_2/subSub)rot_equiv_conv2d_2/strided_slice:output:0!rot_equiv_conv2d_2/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_conv2d_2/clip_by_value/MinimumMinimum!rot_equiv_conv2d_2/Const:output:0rot_equiv_conv2d_2/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_conv2d_2/clip_by_valueMaximum,rot_equiv_conv2d_2/clip_by_value/Minimum:z:0+rot_equiv_conv2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Э
rot_equiv_conv2d_2/GatherV2GatherV2!rot_equiv_pool2d_1/stack:output:0$rot_equiv_conv2d_2/clip_by_value:z:0)rot_equiv_conv2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" г
-rot_equiv_conv2d_2/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0У
rot_equiv_conv2d_2/convolutionConv2D$rot_equiv_conv2d_2/GatherV2:output:05rot_equiv_conv2d_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_1Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_2/strided_slice_1StridedSlice#rot_equiv_conv2d_2/Shape_1:output:01rot_equiv_conv2d_2/strided_slice_1/stack:output:03rot_equiv_conv2d_2/strided_slice_1/stack_1:output:03rot_equiv_conv2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_2/sub_1Sub+rot_equiv_conv2d_2/strided_slice_1:output:0#rot_equiv_conv2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_2/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_2/Const_1:output:0rot_equiv_conv2d_2/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_2/clip_by_value_1Maximum.rot_equiv_conv2d_2/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_2/GatherV2_1GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_1:z:0+rot_equiv_conv2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" «
/rot_equiv_conv2d_2/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Ь
 rot_equiv_conv2d_2/convolution_1Conv2D&rot_equiv_conv2d_2/GatherV2_1:output:07rot_equiv_conv2d_2/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_2Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_2/strided_slice_2StridedSlice#rot_equiv_conv2d_2/Shape_2:output:01rot_equiv_conv2d_2/strided_slice_2/stack:output:03rot_equiv_conv2d_2/strided_slice_2/stack_1:output:03rot_equiv_conv2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_2/sub_2Sub+rot_equiv_conv2d_2/strided_slice_2:output:0#rot_equiv_conv2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_2/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_2/Const_2:output:0rot_equiv_conv2d_2/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_2/clip_by_value_2Maximum.rot_equiv_conv2d_2/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_2/GatherV2_2GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_2:z:0+rot_equiv_conv2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" «
/rot_equiv_conv2d_2/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Ь
 rot_equiv_conv2d_2/convolution_2Conv2D&rot_equiv_conv2d_2/GatherV2_2:output:07rot_equiv_conv2d_2/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_3Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_2/strided_slice_3StridedSlice#rot_equiv_conv2d_2/Shape_3:output:01rot_equiv_conv2d_2/strided_slice_3/stack:output:03rot_equiv_conv2d_2/strided_slice_3/stack_1:output:03rot_equiv_conv2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_2/sub_3Sub+rot_equiv_conv2d_2/strided_slice_3:output:0#rot_equiv_conv2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_2/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_2/Const_3:output:0rot_equiv_conv2d_2/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_2/clip_by_value_3Maximum.rot_equiv_conv2d_2/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_2/GatherV2_3GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_3:z:0+rot_equiv_conv2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" «
/rot_equiv_conv2d_2/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Ь
 rot_equiv_conv2d_2/convolution_3Conv2D&rot_equiv_conv2d_2/GatherV2_3:output:07rot_equiv_conv2d_2/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
д
rot_equiv_conv2d_2/stackPack'rot_equiv_conv2d_2/convolution:output:0)rot_equiv_conv2d_2/convolution_1:output:0)rot_equiv_conv2d_2/convolution_2:output:0)rot_equiv_conv2d_2/convolution_3:output:0*
N*
T0*3
_output_shapes!
:           @*
axis■        ђ
rot_equiv_conv2d_2/ReluRelu!rot_equiv_conv2d_2/stack:output:0*
T0*3
_output_shapes!
:           @ў
)rot_equiv_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0й
rot_equiv_conv2d_2/BiasAddBiasAdd%rot_equiv_conv2d_2/Relu:activations:01rot_equiv_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:           @Z
rot_equiv_pool2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_2/ShapeShape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d_2/strided_sliceStridedSlice!rot_equiv_pool2d_2/Shape:output:0/rot_equiv_pool2d_2/strided_slice/stack:output:01rot_equiv_pool2d_2/strided_slice/stack_1:output:01rot_equiv_pool2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d_2/subSub)rot_equiv_pool2d_2/strided_slice:output:0!rot_equiv_pool2d_2/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d_2/clip_by_value/MinimumMinimum!rot_equiv_pool2d_2/Const:output:0rot_equiv_pool2d_2/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d_2/clip_by_valueMaximum,rot_equiv_pool2d_2/clip_by_value/Minimum:z:0+rot_equiv_pool2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d_2/GatherV2GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0$rot_equiv_pool2d_2/clip_by_value:z:0)rot_equiv_pool2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @╚
*rot_equiv_pool2d_2/max_pooling2d_2/MaxPoolMaxPool$rot_equiv_pool2d_2/GatherV2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_1Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_2/strided_slice_1StridedSlice#rot_equiv_pool2d_2/Shape_1:output:01rot_equiv_pool2d_2/strided_slice_1/stack:output:03rot_equiv_pool2d_2/strided_slice_1/stack_1:output:03rot_equiv_pool2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_2/sub_1Sub+rot_equiv_pool2d_2/strided_slice_1:output:0#rot_equiv_pool2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_2/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_2/Const_1:output:0rot_equiv_pool2d_2/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_2/clip_by_value_1Maximum.rot_equiv_pool2d_2/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_2/GatherV2_1GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_1:z:0+rot_equiv_pool2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @╠
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1MaxPool&rot_equiv_pool2d_2/GatherV2_1:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_2Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_2/strided_slice_2StridedSlice#rot_equiv_pool2d_2/Shape_2:output:01rot_equiv_pool2d_2/strided_slice_2/stack:output:03rot_equiv_pool2d_2/strided_slice_2/stack_1:output:03rot_equiv_pool2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_2/sub_2Sub+rot_equiv_pool2d_2/strided_slice_2:output:0#rot_equiv_pool2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_2/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_2/Const_2:output:0rot_equiv_pool2d_2/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_2/clip_by_value_2Maximum.rot_equiv_pool2d_2/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_2/GatherV2_2GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_2:z:0+rot_equiv_pool2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @╠
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2MaxPool&rot_equiv_pool2d_2/GatherV2_2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_3Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_2/strided_slice_3StridedSlice#rot_equiv_pool2d_2/Shape_3:output:01rot_equiv_pool2d_2/strided_slice_3/stack:output:03rot_equiv_pool2d_2/strided_slice_3/stack_1:output:03rot_equiv_pool2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_2/sub_3Sub+rot_equiv_pool2d_2/strided_slice_3:output:0#rot_equiv_pool2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_2/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_2/Const_3:output:0rot_equiv_pool2d_2/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_2/clip_by_value_3Maximum.rot_equiv_pool2d_2/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_2/GatherV2_3GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_3:z:0+rot_equiv_pool2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @╠
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3MaxPool&rot_equiv_pool2d_2/GatherV2_3:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
о
rot_equiv_pool2d_2/stackPack3rot_equiv_pool2d_2/max_pooling2d_2/MaxPool:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        Z
rot_equiv_conv2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_3/ShapeShape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_conv2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_conv2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_conv2d_3/strided_sliceStridedSlice!rot_equiv_conv2d_3/Shape:output:0/rot_equiv_conv2d_3/strided_slice/stack:output:01rot_equiv_conv2d_3/strided_slice/stack_1:output:01rot_equiv_conv2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_conv2d_3/subSub)rot_equiv_conv2d_3/strided_slice:output:0!rot_equiv_conv2d_3/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_conv2d_3/clip_by_value/MinimumMinimum!rot_equiv_conv2d_3/Const:output:0rot_equiv_conv2d_3/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_conv2d_3/clip_by_valueMaximum,rot_equiv_conv2d_3/clip_by_value/Minimum:z:0+rot_equiv_conv2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Э
rot_equiv_conv2d_3/GatherV2GatherV2!rot_equiv_pool2d_2/stack:output:0$rot_equiv_conv2d_3/clip_by_value:z:0)rot_equiv_conv2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @г
-rot_equiv_conv2d_3/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0У
rot_equiv_conv2d_3/convolutionConv2D$rot_equiv_conv2d_3/GatherV2:output:05rot_equiv_conv2d_3/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_1Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_3/strided_slice_1StridedSlice#rot_equiv_conv2d_3/Shape_1:output:01rot_equiv_conv2d_3/strided_slice_1/stack:output:03rot_equiv_conv2d_3/strided_slice_1/stack_1:output:03rot_equiv_conv2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_3/sub_1Sub+rot_equiv_conv2d_3/strided_slice_1:output:0#rot_equiv_conv2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_3/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_3/Const_1:output:0rot_equiv_conv2d_3/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_3/clip_by_value_1Maximum.rot_equiv_conv2d_3/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_3/GatherV2_1GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_1:z:0+rot_equiv_conv2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @«
/rot_equiv_conv2d_3/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ь
 rot_equiv_conv2d_3/convolution_1Conv2D&rot_equiv_conv2d_3/GatherV2_1:output:07rot_equiv_conv2d_3/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_2Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_3/strided_slice_2StridedSlice#rot_equiv_conv2d_3/Shape_2:output:01rot_equiv_conv2d_3/strided_slice_2/stack:output:03rot_equiv_conv2d_3/strided_slice_2/stack_1:output:03rot_equiv_conv2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_3/sub_2Sub+rot_equiv_conv2d_3/strided_slice_2:output:0#rot_equiv_conv2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_3/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_3/Const_2:output:0rot_equiv_conv2d_3/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_3/clip_by_value_2Maximum.rot_equiv_conv2d_3/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_3/GatherV2_2GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_2:z:0+rot_equiv_conv2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @«
/rot_equiv_conv2d_3/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ь
 rot_equiv_conv2d_3/convolution_2Conv2D&rot_equiv_conv2d_3/GatherV2_2:output:07rot_equiv_conv2d_3/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_3Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_3/strided_slice_3StridedSlice#rot_equiv_conv2d_3/Shape_3:output:01rot_equiv_conv2d_3/strided_slice_3/stack:output:03rot_equiv_conv2d_3/strided_slice_3/stack_1:output:03rot_equiv_conv2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_3/sub_3Sub+rot_equiv_conv2d_3/strided_slice_3:output:0#rot_equiv_conv2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_3/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_3/Const_3:output:0rot_equiv_conv2d_3/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_3/clip_by_value_3Maximum.rot_equiv_conv2d_3/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_3/GatherV2_3GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_3:z:0+rot_equiv_conv2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @«
/rot_equiv_conv2d_3/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ь
 rot_equiv_conv2d_3/convolution_3Conv2D&rot_equiv_conv2d_3/GatherV2_3:output:07rot_equiv_conv2d_3/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
д
rot_equiv_conv2d_3/stackPack'rot_equiv_conv2d_3/convolution:output:0)rot_equiv_conv2d_3/convolution_1:output:0)rot_equiv_conv2d_3/convolution_2:output:0)rot_equiv_conv2d_3/convolution_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        ђ
rot_equiv_conv2d_3/ReluRelu!rot_equiv_conv2d_3/stack:output:0*
T0*3
_output_shapes!
:         @ў
)rot_equiv_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0й
rot_equiv_conv2d_3/BiasAddBiasAdd%rot_equiv_conv2d_3/Relu:activations:01rot_equiv_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         @Z
rot_equiv_pool2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_3/ShapeShape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d_3/strided_sliceStridedSlice!rot_equiv_pool2d_3/Shape:output:0/rot_equiv_pool2d_3/strided_slice/stack:output:01rot_equiv_pool2d_3/strided_slice/stack_1:output:01rot_equiv_pool2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d_3/subSub)rot_equiv_pool2d_3/strided_slice:output:0!rot_equiv_pool2d_3/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d_3/clip_by_value/MinimumMinimum!rot_equiv_pool2d_3/Const:output:0rot_equiv_pool2d_3/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d_3/clip_by_valueMaximum,rot_equiv_pool2d_3/clip_by_value/Minimum:z:0+rot_equiv_pool2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d_3/GatherV2GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0$rot_equiv_pool2d_3/clip_by_value:z:0)rot_equiv_pool2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @╚
*rot_equiv_pool2d_3/max_pooling2d_3/MaxPoolMaxPool$rot_equiv_pool2d_3/GatherV2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_1Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_3/strided_slice_1StridedSlice#rot_equiv_pool2d_3/Shape_1:output:01rot_equiv_pool2d_3/strided_slice_1/stack:output:03rot_equiv_pool2d_3/strided_slice_1/stack_1:output:03rot_equiv_pool2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_3/sub_1Sub+rot_equiv_pool2d_3/strided_slice_1:output:0#rot_equiv_pool2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_3/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_3/Const_1:output:0rot_equiv_pool2d_3/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_3/clip_by_value_1Maximum.rot_equiv_pool2d_3/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_3/GatherV2_1GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_1:z:0+rot_equiv_pool2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @╠
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1MaxPool&rot_equiv_pool2d_3/GatherV2_1:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_2Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_3/strided_slice_2StridedSlice#rot_equiv_pool2d_3/Shape_2:output:01rot_equiv_pool2d_3/strided_slice_2/stack:output:03rot_equiv_pool2d_3/strided_slice_2/stack_1:output:03rot_equiv_pool2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_3/sub_2Sub+rot_equiv_pool2d_3/strided_slice_2:output:0#rot_equiv_pool2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_3/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_3/Const_2:output:0rot_equiv_pool2d_3/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_3/clip_by_value_2Maximum.rot_equiv_pool2d_3/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_3/GatherV2_2GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_2:z:0+rot_equiv_pool2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @╠
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2MaxPool&rot_equiv_pool2d_3/GatherV2_2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_3Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_3/strided_slice_3StridedSlice#rot_equiv_pool2d_3/Shape_3:output:01rot_equiv_pool2d_3/strided_slice_3/stack:output:03rot_equiv_pool2d_3/strided_slice_3/stack_1:output:03rot_equiv_pool2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_3/sub_3Sub+rot_equiv_pool2d_3/strided_slice_3:output:0#rot_equiv_pool2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_3/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_3/Const_3:output:0rot_equiv_pool2d_3/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_3/clip_by_value_3Maximum.rot_equiv_pool2d_3/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_3/GatherV2_3GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_3:z:0+rot_equiv_pool2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @╠
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3MaxPool&rot_equiv_pool2d_3/GatherV2_3:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
о
rot_equiv_pool2d_3/stackPack3rot_equiv_pool2d_3/max_pooling2d_3/MaxPool:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        Z
rot_equiv_conv2d_4/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_4/ShapeShape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_conv2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_conv2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_conv2d_4/strided_sliceStridedSlice!rot_equiv_conv2d_4/Shape:output:0/rot_equiv_conv2d_4/strided_slice/stack:output:01rot_equiv_conv2d_4/strided_slice/stack_1:output:01rot_equiv_conv2d_4/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_4/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_conv2d_4/subSub)rot_equiv_conv2d_4/strided_slice:output:0!rot_equiv_conv2d_4/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_conv2d_4/clip_by_value/MinimumMinimum!rot_equiv_conv2d_4/Const:output:0rot_equiv_conv2d_4/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_4/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_conv2d_4/clip_by_valueMaximum,rot_equiv_conv2d_4/clip_by_value/Minimum:z:0+rot_equiv_conv2d_4/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Э
rot_equiv_conv2d_4/GatherV2GatherV2!rot_equiv_pool2d_3/stack:output:0$rot_equiv_conv2d_4/clip_by_value:z:0)rot_equiv_conv2d_4/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Г
-rot_equiv_conv2d_4/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0ж
rot_equiv_conv2d_4/convolutionConv2D$rot_equiv_conv2d_4/GatherV2:output:05rot_equiv_conv2d_4/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_1Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_4/strided_slice_1StridedSlice#rot_equiv_conv2d_4/Shape_1:output:01rot_equiv_conv2d_4/strided_slice_1/stack:output:03rot_equiv_conv2d_4/strided_slice_1/stack_1:output:03rot_equiv_conv2d_4/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_4/sub_1Sub+rot_equiv_conv2d_4/strided_slice_1:output:0#rot_equiv_conv2d_4/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_4/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_4/Const_1:output:0rot_equiv_conv2d_4/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_4/clip_by_value_1Maximum.rot_equiv_conv2d_4/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_4/GatherV2_1GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_1:z:0+rot_equiv_conv2d_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @»
/rot_equiv_conv2d_4/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0№
 rot_equiv_conv2d_4/convolution_1Conv2D&rot_equiv_conv2d_4/GatherV2_1:output:07rot_equiv_conv2d_4/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_2Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_4/strided_slice_2StridedSlice#rot_equiv_conv2d_4/Shape_2:output:01rot_equiv_conv2d_4/strided_slice_2/stack:output:03rot_equiv_conv2d_4/strided_slice_2/stack_1:output:03rot_equiv_conv2d_4/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_4/sub_2Sub+rot_equiv_conv2d_4/strided_slice_2:output:0#rot_equiv_conv2d_4/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_4/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_4/Const_2:output:0rot_equiv_conv2d_4/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_4/clip_by_value_2Maximum.rot_equiv_conv2d_4/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_4/GatherV2_2GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_2:z:0+rot_equiv_conv2d_4/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @»
/rot_equiv_conv2d_4/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0№
 rot_equiv_conv2d_4/convolution_2Conv2D&rot_equiv_conv2d_4/GatherV2_2:output:07rot_equiv_conv2d_4/convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_3Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_4/strided_slice_3StridedSlice#rot_equiv_conv2d_4/Shape_3:output:01rot_equiv_conv2d_4/strided_slice_3/stack:output:03rot_equiv_conv2d_4/strided_slice_3/stack_1:output:03rot_equiv_conv2d_4/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_4/sub_3Sub+rot_equiv_conv2d_4/strided_slice_3:output:0#rot_equiv_conv2d_4/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_4/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_4/Const_3:output:0rot_equiv_conv2d_4/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_4/clip_by_value_3Maximum.rot_equiv_conv2d_4/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_4/GatherV2_3GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_3:z:0+rot_equiv_conv2d_4/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @»
/rot_equiv_conv2d_4/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0№
 rot_equiv_conv2d_4/convolution_3Conv2D&rot_equiv_conv2d_4/GatherV2_3:output:07rot_equiv_conv2d_4/convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
Д
rot_equiv_conv2d_4/stackPack'rot_equiv_conv2d_4/convolution:output:0)rot_equiv_conv2d_4/convolution_1:output:0)rot_equiv_conv2d_4/convolution_2:output:0)rot_equiv_conv2d_4/convolution_3:output:0*
N*
T0*4
_output_shapes"
 :         ђ*
axis■        Ђ
rot_equiv_conv2d_4/ReluRelu!rot_equiv_conv2d_4/stack:output:0*
T0*4
_output_shapes"
 :         ђЎ
)rot_equiv_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Й
rot_equiv_conv2d_4/BiasAddBiasAdd%rot_equiv_conv2d_4/Relu:activations:01rot_equiv_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         ђm
"rot_inv_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        ц
rot_inv_pool/MaxMax#rot_equiv_conv2d_4/BiasAdd:output:0+rot_inv_pool/Max/reduction_indices:output:0*
T0*0
_output_shapes
:         ђ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  ђ
flatten/ReshapeReshaperot_inv_pool/Max:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         ђЂ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype0Є
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:          ё
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0І
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ѓ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ╬

NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp(^rot_equiv_conv2d/BiasAdd/ReadVariableOp ^rot_equiv_conv2d/ReadVariableOp"^rot_equiv_conv2d/ReadVariableOp_1"^rot_equiv_conv2d/ReadVariableOp_2,^rot_equiv_conv2d/convolution/ReadVariableOp*^rot_equiv_conv2d_1/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_1/convolution/ReadVariableOp0^rot_equiv_conv2d_1/convolution_1/ReadVariableOp0^rot_equiv_conv2d_1/convolution_2/ReadVariableOp0^rot_equiv_conv2d_1/convolution_3/ReadVariableOp*^rot_equiv_conv2d_2/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_2/convolution/ReadVariableOp0^rot_equiv_conv2d_2/convolution_1/ReadVariableOp0^rot_equiv_conv2d_2/convolution_2/ReadVariableOp0^rot_equiv_conv2d_2/convolution_3/ReadVariableOp*^rot_equiv_conv2d_3/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_3/convolution/ReadVariableOp0^rot_equiv_conv2d_3/convolution_1/ReadVariableOp0^rot_equiv_conv2d_3/convolution_2/ReadVariableOp0^rot_equiv_conv2d_3/convolution_3/ReadVariableOp*^rot_equiv_conv2d_4/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_4/convolution/ReadVariableOp0^rot_equiv_conv2d_4/convolution_1/ReadVariableOp0^rot_equiv_conv2d_4/convolution_2/ReadVariableOp0^rot_equiv_conv2d_4/convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2R
'rot_equiv_conv2d/BiasAdd/ReadVariableOp'rot_equiv_conv2d/BiasAdd/ReadVariableOp2B
rot_equiv_conv2d/ReadVariableOprot_equiv_conv2d/ReadVariableOp2F
!rot_equiv_conv2d/ReadVariableOp_1!rot_equiv_conv2d/ReadVariableOp_12F
!rot_equiv_conv2d/ReadVariableOp_2!rot_equiv_conv2d/ReadVariableOp_22Z
+rot_equiv_conv2d/convolution/ReadVariableOp+rot_equiv_conv2d/convolution/ReadVariableOp2V
)rot_equiv_conv2d_1/BiasAdd/ReadVariableOp)rot_equiv_conv2d_1/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_1/convolution/ReadVariableOp-rot_equiv_conv2d_1/convolution/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_1/ReadVariableOp/rot_equiv_conv2d_1/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_2/ReadVariableOp/rot_equiv_conv2d_1/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_3/ReadVariableOp/rot_equiv_conv2d_1/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_2/BiasAdd/ReadVariableOp)rot_equiv_conv2d_2/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_2/convolution/ReadVariableOp-rot_equiv_conv2d_2/convolution/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_1/ReadVariableOp/rot_equiv_conv2d_2/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_2/ReadVariableOp/rot_equiv_conv2d_2/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_3/ReadVariableOp/rot_equiv_conv2d_2/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_3/BiasAdd/ReadVariableOp)rot_equiv_conv2d_3/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_3/convolution/ReadVariableOp-rot_equiv_conv2d_3/convolution/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_1/ReadVariableOp/rot_equiv_conv2d_3/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_2/ReadVariableOp/rot_equiv_conv2d_3/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_3/ReadVariableOp/rot_equiv_conv2d_3/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_4/BiasAdd/ReadVariableOp)rot_equiv_conv2d_4/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_4/convolution/ReadVariableOp-rot_equiv_conv2d_4/convolution/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_1/ReadVariableOp/rot_equiv_conv2d_4/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_2/ReadVariableOp/rot_equiv_conv2d_4/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_3/ReadVariableOp/rot_equiv_conv2d_4/convolution_3/ReadVariableOp:Y U
1
_output_shapes
:         љљ
 
_user_specified_nameinputs
С
O
3__inference_rot_equiv_pool2d_1_layer_call_fn_372496

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         "" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_370198l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:         "" "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         EE :[ W
3
_output_shapes!
:         EE 
 
_user_specified_nameinputs
Љ
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_369867

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
к6
j
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_370337

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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @б
max_pooling2d_2/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @д
max_pooling2d_2/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @д
max_pooling2d_2/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @д
max_pooling2d_2/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
э
stackPack max_pooling2d_2/MaxPool:output:0"max_pooling2d_2/MaxPool_1:output:0"max_pooling2d_2/MaxPool_2:output:0"max_pooling2d_2/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           @:[ W
3
_output_shapes!
:           @
 
_user_specified_nameinputs
к	
З
C__inference_dense_1_layer_call_and_return_conditional_losses_370597

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
С
O
3__inference_rot_equiv_pool2d_2_layer_call_fn_372641

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_370337l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           @:[ W
3
_output_shapes!
:           @
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_369891

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Х
џ
+__inference_sequential_layer_call_fn_370635
rot_equiv_conv2d_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@ђ
	unknown_8:	ђ
	unknown_9:	ђ 

unknown_10: 

unknown_11: 

unknown_12:
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_370604o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
1
_output_shapes
:         љљ
0
_user_specified_namerot_equiv_conv2d_input
к6
j
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_370476

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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @б
max_pooling2d_3/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @д
max_pooling2d_3/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @д
max_pooling2d_3/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @д
max_pooling2d_3/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
э
stackPack max_pooling2d_3/MaxPool:output:0"max_pooling2d_3/MaxPool_1:output:0"max_pooling2d_3/MaxPool_2:output:0"max_pooling2d_3/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
є
і
+__inference_sequential_layer_call_fn_371082

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@ђ
	unknown_8:	ђ
	unknown_9:	ђ 

unknown_10: 

unknown_11: 

unknown_12:
identityѕбStatefulPartitionedCall§
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
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_370821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         љљ
 
_user_specified_nameinputs
К
_
C__inference_flatten_layer_call_and_return_conditional_losses_372948

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
тC
Ь
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_370409

inputs=
#convolution_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconvolution/ReadVariableOpбconvolution_1/ReadVariableOpбconvolution_2/ReadVariableOpбconvolution_3/ReadVariableOpG
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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @є
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0»
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @ѕ
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0х
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @ѕ
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0х
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @ѕ
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0х
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
К
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ё
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         @k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:         @┘
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
ЪП
Ы
!__inference__wrapped_model_369858
rot_equiv_conv2d_inputY
?sequential_rot_equiv_conv2d_convolution_readvariableop_resource: I
;sequential_rot_equiv_conv2d_biasadd_readvariableop_resource: [
Asequential_rot_equiv_conv2d_1_convolution_readvariableop_resource:  K
=sequential_rot_equiv_conv2d_1_biasadd_readvariableop_resource: [
Asequential_rot_equiv_conv2d_2_convolution_readvariableop_resource: @K
=sequential_rot_equiv_conv2d_2_biasadd_readvariableop_resource:@[
Asequential_rot_equiv_conv2d_3_convolution_readvariableop_resource:@@K
=sequential_rot_equiv_conv2d_3_biasadd_readvariableop_resource:@\
Asequential_rot_equiv_conv2d_4_convolution_readvariableop_resource:@ђL
=sequential_rot_equiv_conv2d_4_biasadd_readvariableop_resource:	ђB
/sequential_dense_matmul_readvariableop_resource:	ђ >
0sequential_dense_biasadd_readvariableop_resource: C
1sequential_dense_1_matmul_readvariableop_resource: @
2sequential_dense_1_biasadd_readvariableop_resource:
identityѕб'sequential/dense/BiasAdd/ReadVariableOpб&sequential/dense/MatMul/ReadVariableOpб)sequential/dense_1/BiasAdd/ReadVariableOpб(sequential/dense_1/MatMul/ReadVariableOpб2sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOpб*sequential/rot_equiv_conv2d/ReadVariableOpб,sequential/rot_equiv_conv2d/ReadVariableOp_1б,sequential/rot_equiv_conv2d/ReadVariableOp_2б6sequential/rot_equiv_conv2d/convolution/ReadVariableOpб4sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOpб8sequential/rot_equiv_conv2d_1/convolution/ReadVariableOpб:sequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOpб:sequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOpб:sequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOpб4sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOpб8sequential/rot_equiv_conv2d_2/convolution/ReadVariableOpб:sequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOpб:sequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOpб:sequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOpб4sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOpб8sequential/rot_equiv_conv2d_3/convolution/ReadVariableOpб:sequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOpб:sequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOpб:sequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOpб4sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOpб8sequential/rot_equiv_conv2d_4/convolution/ReadVariableOpб:sequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOpб:sequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOpб:sequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOpЙ
6sequential/rot_equiv_conv2d/convolution/ReadVariableOpReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Ь
'sequential/rot_equiv_conv2d/convolutionConv2Drot_equiv_conv2d_input>sequential/rot_equiv_conv2d/convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
и
/sequential/rot_equiv_conv2d/Rank/ReadVariableOpReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0b
 sequential/rot_equiv_conv2d/RankConst*
_output_shapes
: *
dtype0*
value	B :i
'sequential/rot_equiv_conv2d/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'sequential/rot_equiv_conv2d/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Н
!sequential/rot_equiv_conv2d/rangeRange0sequential/rot_equiv_conv2d/range/start:output:0)sequential/rot_equiv_conv2d/Rank:output:00sequential/rot_equiv_conv2d/range/delta:output:0*
_output_shapes
:љ
7sequential/rot_equiv_conv2d/TensorScatterUpdate/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       ѕ
7sequential/rot_equiv_conv2d/TensorScatterUpdate/updatesConst*
_output_shapes
:*
dtype0*
valueB"       Ф
/sequential/rot_equiv_conv2d/TensorScatterUpdateTensorScatterUpdate*sequential/rot_equiv_conv2d/range:output:0@sequential/rot_equiv_conv2d/TensorScatterUpdate/indices:output:0@sequential/rot_equiv_conv2d/TensorScatterUpdate/updates:output:0*
T0*
Tindices0*
_output_shapes
:▓
*sequential/rot_equiv_conv2d/ReadVariableOpReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0d
"sequential/rot_equiv_conv2d/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :t
*sequential/rot_equiv_conv2d/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:╠
%sequential/rot_equiv_conv2d/ReverseV2	ReverseV22sequential/rot_equiv_conv2d/ReadVariableOp:value:03sequential/rot_equiv_conv2d/ReverseV2/axis:output:0*
T0*&
_output_shapes
: ═
%sequential/rot_equiv_conv2d/transpose	Transpose.sequential/rot_equiv_conv2d/ReverseV2:output:08sequential/rot_equiv_conv2d/TensorScatterUpdate:output:0*
T0*&
_output_shapes
: █
)sequential/rot_equiv_conv2d/convolution_1Conv2Drot_equiv_conv2d_input)sequential/rot_equiv_conv2d/transpose:y:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
d
"sequential/rot_equiv_conv2d/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential/rot_equiv_conv2d/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : k
)sequential/rot_equiv_conv2d/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :П
#sequential/rot_equiv_conv2d/range_1Range2sequential/rot_equiv_conv2d/range_1/start:output:0+sequential/rot_equiv_conv2d/Rank_2:output:02sequential/rot_equiv_conv2d/range_1/delta:output:0*
_output_shapes
:њ
9sequential/rot_equiv_conv2d/TensorScatterUpdate_1/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      і
9sequential/rot_equiv_conv2d/TensorScatterUpdate_1/updatesConst*
_output_shapes
:*
dtype0*
valueB"      │
1sequential/rot_equiv_conv2d/TensorScatterUpdate_1TensorScatterUpdate,sequential/rot_equiv_conv2d/range_1:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_1/indices:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:Я
'sequential/rot_equiv_conv2d/transpose_1	Transpose2sequential/rot_equiv_conv2d/convolution_1:output:0:sequential/rot_equiv_conv2d/TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:         јј d
"sequential/rot_equiv_conv2d/Rank_3Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:н
'sequential/rot_equiv_conv2d/ReverseV2_1	ReverseV2+sequential/rot_equiv_conv2d/transpose_1:y:05sequential/rot_equiv_conv2d/ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:         јј ╣
1sequential/rot_equiv_conv2d/Rank_4/ReadVariableOpReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0d
"sequential/rot_equiv_conv2d/Rank_4Const*
_output_shapes
: *
dtype0*
value	B :┤
,sequential/rot_equiv_conv2d/ReadVariableOp_1ReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0d
"sequential/rot_equiv_conv2d/Rank_5Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_2/axisConst*
_output_shapes
:*
dtype0*
valueB: м
'sequential/rot_equiv_conv2d/ReverseV2_2	ReverseV24sequential/rot_equiv_conv2d/ReadVariableOp_1:value:05sequential/rot_equiv_conv2d/ReverseV2_2/axis:output:0*
T0*&
_output_shapes
: d
"sequential/rot_equiv_conv2d/Rank_6Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_3/axisConst*
_output_shapes
:*
dtype0*
valueB:╬
'sequential/rot_equiv_conv2d/ReverseV2_3	ReverseV20sequential/rot_equiv_conv2d/ReverseV2_2:output:05sequential/rot_equiv_conv2d/ReverseV2_3/axis:output:0*
T0*&
_output_shapes
: Р
)sequential/rot_equiv_conv2d/convolution_2Conv2Drot_equiv_conv2d_input0sequential/rot_equiv_conv2d/ReverseV2_3:output:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
d
"sequential/rot_equiv_conv2d/Rank_7Const*
_output_shapes
: *
dtype0*
value	B :d
"sequential/rot_equiv_conv2d/Rank_8Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_4/axisConst*
_output_shapes
:*
dtype0*
valueB:█
'sequential/rot_equiv_conv2d/ReverseV2_4	ReverseV22sequential/rot_equiv_conv2d/convolution_2:output:05sequential/rot_equiv_conv2d/ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:         јј d
"sequential/rot_equiv_conv2d/Rank_9Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:┘
'sequential/rot_equiv_conv2d/ReverseV2_5	ReverseV20sequential/rot_equiv_conv2d/ReverseV2_4:output:05sequential/rot_equiv_conv2d/ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:         јј ║
2sequential/rot_equiv_conv2d/Rank_10/ReadVariableOpReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0e
#sequential/rot_equiv_conv2d/Rank_10Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential/rot_equiv_conv2d/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : k
)sequential/rot_equiv_conv2d/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :я
#sequential/rot_equiv_conv2d/range_2Range2sequential/rot_equiv_conv2d/range_2/start:output:0,sequential/rot_equiv_conv2d/Rank_10:output:02sequential/rot_equiv_conv2d/range_2/delta:output:0*
_output_shapes
:њ
9sequential/rot_equiv_conv2d/TensorScatterUpdate_2/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       і
9sequential/rot_equiv_conv2d/TensorScatterUpdate_2/updatesConst*
_output_shapes
:*
dtype0*
valueB"       │
1sequential/rot_equiv_conv2d/TensorScatterUpdate_2TensorScatterUpdate,sequential/rot_equiv_conv2d/range_2:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_2/indices:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:┤
,sequential/rot_equiv_conv2d/ReadVariableOp_2ReadVariableOp?sequential_rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0О
'sequential/rot_equiv_conv2d/transpose_2	Transpose4sequential/rot_equiv_conv2d/ReadVariableOp_2:value:0:sequential/rot_equiv_conv2d/TensorScatterUpdate_2:output:0*
T0*&
_output_shapes
: e
#sequential/rot_equiv_conv2d/Rank_11Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_6/axisConst*
_output_shapes
:*
dtype0*
valueB:╔
'sequential/rot_equiv_conv2d/ReverseV2_6	ReverseV2+sequential/rot_equiv_conv2d/transpose_2:y:05sequential/rot_equiv_conv2d/ReverseV2_6/axis:output:0*
T0*&
_output_shapes
: Р
)sequential/rot_equiv_conv2d/convolution_3Conv2Drot_equiv_conv2d_input0sequential/rot_equiv_conv2d/ReverseV2_6:output:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
e
#sequential/rot_equiv_conv2d/Rank_12Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential/rot_equiv_conv2d/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : k
)sequential/rot_equiv_conv2d/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :я
#sequential/rot_equiv_conv2d/range_3Range2sequential/rot_equiv_conv2d/range_3/start:output:0,sequential/rot_equiv_conv2d/Rank_12:output:02sequential/rot_equiv_conv2d/range_3/delta:output:0*
_output_shapes
:њ
9sequential/rot_equiv_conv2d/TensorScatterUpdate_3/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      і
9sequential/rot_equiv_conv2d/TensorScatterUpdate_3/updatesConst*
_output_shapes
:*
dtype0*
valueB"      │
1sequential/rot_equiv_conv2d/TensorScatterUpdate_3TensorScatterUpdate,sequential/rot_equiv_conv2d/range_3:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_3/indices:output:0Bsequential/rot_equiv_conv2d/TensorScatterUpdate_3/updates:output:0*
T0*
Tindices0*
_output_shapes
:e
#sequential/rot_equiv_conv2d/Rank_13Const*
_output_shapes
: *
dtype0*
value	B :v
,sequential/rot_equiv_conv2d/ReverseV2_7/axisConst*
_output_shapes
:*
dtype0*
valueB:█
'sequential/rot_equiv_conv2d/ReverseV2_7	ReverseV22sequential/rot_equiv_conv2d/convolution_3:output:05sequential/rot_equiv_conv2d/ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:         јј я
'sequential/rot_equiv_conv2d/transpose_3	Transpose0sequential/rot_equiv_conv2d/ReverseV2_7:output:0:sequential/rot_equiv_conv2d/TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:         јј ╩
!sequential/rot_equiv_conv2d/stackPack0sequential/rot_equiv_conv2d/convolution:output:00sequential/rot_equiv_conv2d/ReverseV2_1:output:00sequential/rot_equiv_conv2d/ReverseV2_5:output:0+sequential/rot_equiv_conv2d/transpose_3:y:0*
N*
T0*5
_output_shapes#
!:         јј *
axis■        ћ
 sequential/rot_equiv_conv2d/ReluRelu*sequential/rot_equiv_conv2d/stack:output:0*
T0*5
_output_shapes#
!:         јј ф
2sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOpReadVariableOp;sequential_rot_equiv_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┌
#sequential/rot_equiv_conv2d/BiasAddBiasAdd.sequential/rot_equiv_conv2d/Relu:activations:0:sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:         јј c
!sequential/rot_equiv_pool2d/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ї
!sequential/rot_equiv_pool2d/ShapeShape,sequential/rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	ѓ
/sequential/rot_equiv_pool2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ё
1sequential/rot_equiv_pool2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         {
1sequential/rot_equiv_pool2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
)sequential/rot_equiv_pool2d/strided_sliceStridedSlice*sequential/rot_equiv_pool2d/Shape:output:08sequential/rot_equiv_pool2d/strided_slice/stack:output:0:sequential/rot_equiv_pool2d/strided_slice/stack_1:output:0:sequential/rot_equiv_pool2d/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskc
!sequential/rot_equiv_pool2d/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RД
sequential/rot_equiv_pool2d/subSub2sequential/rot_equiv_pool2d/strided_slice:output:0*sequential/rot_equiv_pool2d/sub/y:output:0*
T0	*
_output_shapes
: «
1sequential/rot_equiv_pool2d/clip_by_value/MinimumMinimum*sequential/rot_equiv_pool2d/Const:output:0#sequential/rot_equiv_pool2d/sub:z:0*
T0	*
_output_shapes
: m
+sequential/rot_equiv_pool2d/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ┬
)sequential/rot_equiv_pool2d/clip_by_valueMaximum5sequential/rot_equiv_pool2d/clip_by_value/Minimum:z:04sequential/rot_equiv_pool2d/clip_by_value/y:output:0*
T0	*
_output_shapes
: t
)sequential/rot_equiv_pool2d/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        а
$sequential/rot_equiv_pool2d/GatherV2GatherV2,sequential/rot_equiv_conv2d/BiasAdd:output:0-sequential/rot_equiv_pool2d/clip_by_value:z:02sequential/rot_equiv_pool2d/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј п
1sequential/rot_equiv_pool2d/max_pooling2d/MaxPoolMaxPool-sequential/rot_equiv_pool2d/GatherV2:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
e
#sequential/rot_equiv_pool2d/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RЈ
#sequential/rot_equiv_pool2d/Shape_1Shape,sequential/rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	ё
1sequential/rot_equiv_pool2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        є
3sequential/rot_equiv_pool2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         }
3sequential/rot_equiv_pool2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
+sequential/rot_equiv_pool2d/strided_slice_1StridedSlice,sequential/rot_equiv_pool2d/Shape_1:output:0:sequential/rot_equiv_pool2d/strided_slice_1/stack:output:0<sequential/rot_equiv_pool2d/strided_slice_1/stack_1:output:0<sequential/rot_equiv_pool2d/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 RГ
!sequential/rot_equiv_pool2d/sub_1Sub4sequential/rot_equiv_pool2d/strided_slice_1:output:0,sequential/rot_equiv_pool2d/sub_1/y:output:0*
T0	*
_output_shapes
: ┤
3sequential/rot_equiv_pool2d/clip_by_value_1/MinimumMinimum,sequential/rot_equiv_pool2d/Const_1:output:0%sequential/rot_equiv_pool2d/sub_1:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╚
+sequential/rot_equiv_pool2d/clip_by_value_1Maximum7sequential/rot_equiv_pool2d/clip_by_value_1/Minimum:z:06sequential/rot_equiv_pool2d/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        д
&sequential/rot_equiv_pool2d/GatherV2_1GatherV2,sequential/rot_equiv_conv2d/BiasAdd:output:0/sequential/rot_equiv_pool2d/clip_by_value_1:z:04sequential/rot_equiv_pool2d/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ▄
3sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_1MaxPool/sequential/rot_equiv_pool2d/GatherV2_1:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
e
#sequential/rot_equiv_pool2d/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RЈ
#sequential/rot_equiv_pool2d/Shape_2Shape,sequential/rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	ё
1sequential/rot_equiv_pool2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        є
3sequential/rot_equiv_pool2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         }
3sequential/rot_equiv_pool2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
+sequential/rot_equiv_pool2d/strided_slice_2StridedSlice,sequential/rot_equiv_pool2d/Shape_2:output:0:sequential/rot_equiv_pool2d/strided_slice_2/stack:output:0<sequential/rot_equiv_pool2d/strided_slice_2/stack_1:output:0<sequential/rot_equiv_pool2d/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 RГ
!sequential/rot_equiv_pool2d/sub_2Sub4sequential/rot_equiv_pool2d/strided_slice_2:output:0,sequential/rot_equiv_pool2d/sub_2/y:output:0*
T0	*
_output_shapes
: ┤
3sequential/rot_equiv_pool2d/clip_by_value_2/MinimumMinimum,sequential/rot_equiv_pool2d/Const_2:output:0%sequential/rot_equiv_pool2d/sub_2:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╚
+sequential/rot_equiv_pool2d/clip_by_value_2Maximum7sequential/rot_equiv_pool2d/clip_by_value_2/Minimum:z:06sequential/rot_equiv_pool2d/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        д
&sequential/rot_equiv_pool2d/GatherV2_2GatherV2,sequential/rot_equiv_conv2d/BiasAdd:output:0/sequential/rot_equiv_pool2d/clip_by_value_2:z:04sequential/rot_equiv_pool2d/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ▄
3sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_2MaxPool/sequential/rot_equiv_pool2d/GatherV2_2:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
e
#sequential/rot_equiv_pool2d/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RЈ
#sequential/rot_equiv_pool2d/Shape_3Shape,sequential/rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	ё
1sequential/rot_equiv_pool2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        є
3sequential/rot_equiv_pool2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         }
3sequential/rot_equiv_pool2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
+sequential/rot_equiv_pool2d/strided_slice_3StridedSlice,sequential/rot_equiv_pool2d/Shape_3:output:0:sequential/rot_equiv_pool2d/strided_slice_3/stack:output:0<sequential/rot_equiv_pool2d/strided_slice_3/stack_1:output:0<sequential/rot_equiv_pool2d/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RГ
!sequential/rot_equiv_pool2d/sub_3Sub4sequential/rot_equiv_pool2d/strided_slice_3:output:0,sequential/rot_equiv_pool2d/sub_3/y:output:0*
T0	*
_output_shapes
: ┤
3sequential/rot_equiv_pool2d/clip_by_value_3/MinimumMinimum,sequential/rot_equiv_pool2d/Const_3:output:0%sequential/rot_equiv_pool2d/sub_3:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╚
+sequential/rot_equiv_pool2d/clip_by_value_3Maximum7sequential/rot_equiv_pool2d/clip_by_value_3/Minimum:z:06sequential/rot_equiv_pool2d/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        д
&sequential/rot_equiv_pool2d/GatherV2_3GatherV2,sequential/rot_equiv_conv2d/BiasAdd:output:0/sequential/rot_equiv_pool2d/clip_by_value_3:z:04sequential/rot_equiv_pool2d/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ▄
3sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_3MaxPool/sequential/rot_equiv_pool2d/GatherV2_3:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
ч
!sequential/rot_equiv_pool2d/stackPack:sequential/rot_equiv_pool2d/max_pooling2d/MaxPool:output:0<sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_1:output:0<sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_2:output:0<sequential/rot_equiv_pool2d/max_pooling2d/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         GG *
axis■        e
#sequential/rot_equiv_conv2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ї
#sequential/rot_equiv_conv2d_1/ShapeShape*sequential/rot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	ё
1sequential/rot_equiv_conv2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        є
3sequential/rot_equiv_conv2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         }
3sequential/rot_equiv_conv2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
+sequential/rot_equiv_conv2d_1/strided_sliceStridedSlice,sequential/rot_equiv_conv2d_1/Shape:output:0:sequential/rot_equiv_conv2d_1/strided_slice/stack:output:0<sequential/rot_equiv_conv2d_1/strided_slice/stack_1:output:0<sequential/rot_equiv_conv2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_conv2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RГ
!sequential/rot_equiv_conv2d_1/subSub4sequential/rot_equiv_conv2d_1/strided_slice:output:0,sequential/rot_equiv_conv2d_1/sub/y:output:0*
T0	*
_output_shapes
: ┤
3sequential/rot_equiv_conv2d_1/clip_by_value/MinimumMinimum,sequential/rot_equiv_conv2d_1/Const:output:0%sequential/rot_equiv_conv2d_1/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_conv2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╚
+sequential/rot_equiv_conv2d_1/clip_by_valueMaximum7sequential/rot_equiv_conv2d_1/clip_by_value/Minimum:z:06sequential/rot_equiv_conv2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_conv2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        б
&sequential/rot_equiv_conv2d_1/GatherV2GatherV2*sequential/rot_equiv_pool2d/stack:output:0/sequential/rot_equiv_conv2d_1/clip_by_value:z:04sequential/rot_equiv_conv2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG ┬
8sequential/rot_equiv_conv2d_1/convolution/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Ѕ
)sequential/rot_equiv_conv2d_1/convolutionConv2D/sequential/rot_equiv_conv2d_1/GatherV2:output:0@sequential/rot_equiv_conv2d_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RЈ
%sequential/rot_equiv_conv2d_1/Shape_1Shape*sequential/rot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_1/strided_slice_1StridedSlice.sequential/rot_equiv_conv2d_1/Shape_1:output:0<sequential/rot_equiv_conv2d_1/strided_slice_1/stack:output:0>sequential/rot_equiv_conv2d_1/strided_slice_1/stack_1:output:0>sequential/rot_equiv_conv2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_1/sub_1Sub6sequential/rot_equiv_conv2d_1/strided_slice_1:output:0.sequential/rot_equiv_conv2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_1/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_conv2d_1/Const_1:output:0'sequential/rot_equiv_conv2d_1/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_1/clip_by_value_1Maximum9sequential/rot_equiv_conv2d_1/clip_by_value_1/Minimum:z:08sequential/rot_equiv_conv2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        е
(sequential/rot_equiv_conv2d_1/GatherV2_1GatherV2*sequential/rot_equiv_pool2d/stack:output:01sequential/rot_equiv_conv2d_1/clip_by_value_1:z:06sequential/rot_equiv_conv2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG ─
:sequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Ј
+sequential/rot_equiv_conv2d_1/convolution_1Conv2D1sequential/rot_equiv_conv2d_1/GatherV2_1:output:0Bsequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RЈ
%sequential/rot_equiv_conv2d_1/Shape_2Shape*sequential/rot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_1/strided_slice_2StridedSlice.sequential/rot_equiv_conv2d_1/Shape_2:output:0<sequential/rot_equiv_conv2d_1/strided_slice_2/stack:output:0>sequential/rot_equiv_conv2d_1/strided_slice_2/stack_1:output:0>sequential/rot_equiv_conv2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_1/sub_2Sub6sequential/rot_equiv_conv2d_1/strided_slice_2:output:0.sequential/rot_equiv_conv2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_1/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_conv2d_1/Const_2:output:0'sequential/rot_equiv_conv2d_1/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_1/clip_by_value_2Maximum9sequential/rot_equiv_conv2d_1/clip_by_value_2/Minimum:z:08sequential/rot_equiv_conv2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        е
(sequential/rot_equiv_conv2d_1/GatherV2_2GatherV2*sequential/rot_equiv_pool2d/stack:output:01sequential/rot_equiv_conv2d_1/clip_by_value_2:z:06sequential/rot_equiv_conv2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG ─
:sequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Ј
+sequential/rot_equiv_conv2d_1/convolution_2Conv2D1sequential/rot_equiv_conv2d_1/GatherV2_2:output:0Bsequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RЈ
%sequential/rot_equiv_conv2d_1/Shape_3Shape*sequential/rot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_1/strided_slice_3StridedSlice.sequential/rot_equiv_conv2d_1/Shape_3:output:0<sequential/rot_equiv_conv2d_1/strided_slice_3/stack:output:0>sequential/rot_equiv_conv2d_1/strided_slice_3/stack_1:output:0>sequential/rot_equiv_conv2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_1/sub_3Sub6sequential/rot_equiv_conv2d_1/strided_slice_3:output:0.sequential/rot_equiv_conv2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_1/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_conv2d_1/Const_3:output:0'sequential/rot_equiv_conv2d_1/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_1/clip_by_value_3Maximum9sequential/rot_equiv_conv2d_1/clip_by_value_3/Minimum:z:08sequential/rot_equiv_conv2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        е
(sequential/rot_equiv_conv2d_1/GatherV2_3GatherV2*sequential/rot_equiv_pool2d/stack:output:01sequential/rot_equiv_conv2d_1/clip_by_value_3:z:06sequential/rot_equiv_conv2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG ─
:sequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Ј
+sequential/rot_equiv_conv2d_1/convolution_3Conv2D1sequential/rot_equiv_conv2d_1/GatherV2_3:output:0Bsequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
П
#sequential/rot_equiv_conv2d_1/stackPack2sequential/rot_equiv_conv2d_1/convolution:output:04sequential/rot_equiv_conv2d_1/convolution_1:output:04sequential/rot_equiv_conv2d_1/convolution_2:output:04sequential/rot_equiv_conv2d_1/convolution_3:output:0*
N*
T0*3
_output_shapes!
:         EE *
axis■        ќ
"sequential/rot_equiv_conv2d_1/ReluRelu,sequential/rot_equiv_conv2d_1/stack:output:0*
T0*3
_output_shapes!
:         EE «
4sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp=sequential_rot_equiv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0я
%sequential/rot_equiv_conv2d_1/BiasAddBiasAdd0sequential/rot_equiv_conv2d_1/Relu:activations:0<sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         EE e
#sequential/rot_equiv_pool2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Љ
#sequential/rot_equiv_pool2d_1/ShapeShape.sequential/rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	ё
1sequential/rot_equiv_pool2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        є
3sequential/rot_equiv_pool2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         }
3sequential/rot_equiv_pool2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
+sequential/rot_equiv_pool2d_1/strided_sliceStridedSlice,sequential/rot_equiv_pool2d_1/Shape:output:0:sequential/rot_equiv_pool2d_1/strided_slice/stack:output:0<sequential/rot_equiv_pool2d_1/strided_slice/stack_1:output:0<sequential/rot_equiv_pool2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RГ
!sequential/rot_equiv_pool2d_1/subSub4sequential/rot_equiv_pool2d_1/strided_slice:output:0,sequential/rot_equiv_pool2d_1/sub/y:output:0*
T0	*
_output_shapes
: ┤
3sequential/rot_equiv_pool2d_1/clip_by_value/MinimumMinimum,sequential/rot_equiv_pool2d_1/Const:output:0%sequential/rot_equiv_pool2d_1/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╚
+sequential/rot_equiv_pool2d_1/clip_by_valueMaximum7sequential/rot_equiv_pool2d_1/clip_by_value/Minimum:z:06sequential/rot_equiv_pool2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        д
&sequential/rot_equiv_pool2d_1/GatherV2GatherV2.sequential/rot_equiv_conv2d_1/BiasAdd:output:0/sequential/rot_equiv_pool2d_1/clip_by_value:z:04sequential/rot_equiv_pool2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE я
5sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPoolMaxPool/sequential/rot_equiv_pool2d_1/GatherV2:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RЊ
%sequential/rot_equiv_pool2d_1/Shape_1Shape.sequential/rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_pool2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_pool2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_pool2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_pool2d_1/strided_slice_1StridedSlice.sequential/rot_equiv_pool2d_1/Shape_1:output:0<sequential/rot_equiv_pool2d_1/strided_slice_1/stack:output:0>sequential/rot_equiv_pool2d_1/strided_slice_1/stack_1:output:0>sequential/rot_equiv_pool2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_pool2d_1/sub_1Sub6sequential/rot_equiv_pool2d_1/strided_slice_1:output:0.sequential/rot_equiv_pool2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_pool2d_1/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_pool2d_1/Const_1:output:0'sequential/rot_equiv_pool2d_1/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_pool2d_1/clip_by_value_1Maximum9sequential/rot_equiv_pool2d_1/clip_by_value_1/Minimum:z:08sequential/rot_equiv_pool2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        г
(sequential/rot_equiv_pool2d_1/GatherV2_1GatherV2.sequential/rot_equiv_conv2d_1/BiasAdd:output:01sequential/rot_equiv_pool2d_1/clip_by_value_1:z:06sequential/rot_equiv_pool2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE Р
7sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1MaxPool1sequential/rot_equiv_pool2d_1/GatherV2_1:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RЊ
%sequential/rot_equiv_pool2d_1/Shape_2Shape.sequential/rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_pool2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_pool2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_pool2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_pool2d_1/strided_slice_2StridedSlice.sequential/rot_equiv_pool2d_1/Shape_2:output:0<sequential/rot_equiv_pool2d_1/strided_slice_2/stack:output:0>sequential/rot_equiv_pool2d_1/strided_slice_2/stack_1:output:0>sequential/rot_equiv_pool2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_pool2d_1/sub_2Sub6sequential/rot_equiv_pool2d_1/strided_slice_2:output:0.sequential/rot_equiv_pool2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_pool2d_1/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_pool2d_1/Const_2:output:0'sequential/rot_equiv_pool2d_1/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_pool2d_1/clip_by_value_2Maximum9sequential/rot_equiv_pool2d_1/clip_by_value_2/Minimum:z:08sequential/rot_equiv_pool2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        г
(sequential/rot_equiv_pool2d_1/GatherV2_2GatherV2.sequential/rot_equiv_conv2d_1/BiasAdd:output:01sequential/rot_equiv_pool2d_1/clip_by_value_2:z:06sequential/rot_equiv_pool2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE Р
7sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2MaxPool1sequential/rot_equiv_pool2d_1/GatherV2_2:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RЊ
%sequential/rot_equiv_pool2d_1/Shape_3Shape.sequential/rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_pool2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_pool2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_pool2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_pool2d_1/strided_slice_3StridedSlice.sequential/rot_equiv_pool2d_1/Shape_3:output:0<sequential/rot_equiv_pool2d_1/strided_slice_3/stack:output:0>sequential/rot_equiv_pool2d_1/strided_slice_3/stack_1:output:0>sequential/rot_equiv_pool2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_pool2d_1/sub_3Sub6sequential/rot_equiv_pool2d_1/strided_slice_3:output:0.sequential/rot_equiv_pool2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_pool2d_1/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_pool2d_1/Const_3:output:0'sequential/rot_equiv_pool2d_1/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_pool2d_1/clip_by_value_3Maximum9sequential/rot_equiv_pool2d_1/clip_by_value_3/Minimum:z:08sequential/rot_equiv_pool2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        г
(sequential/rot_equiv_pool2d_1/GatherV2_3GatherV2.sequential/rot_equiv_conv2d_1/BiasAdd:output:01sequential/rot_equiv_pool2d_1/clip_by_value_3:z:06sequential/rot_equiv_pool2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE Р
7sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3MaxPool1sequential/rot_equiv_pool2d_1/GatherV2_3:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
Ї
#sequential/rot_equiv_pool2d_1/stackPack>sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool:output:0@sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1:output:0@sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2:output:0@sequential/rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         "" *
axis■        e
#sequential/rot_equiv_conv2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ј
#sequential/rot_equiv_conv2d_2/ShapeShape,sequential/rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	ё
1sequential/rot_equiv_conv2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        є
3sequential/rot_equiv_conv2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         }
3sequential/rot_equiv_conv2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
+sequential/rot_equiv_conv2d_2/strided_sliceStridedSlice,sequential/rot_equiv_conv2d_2/Shape:output:0:sequential/rot_equiv_conv2d_2/strided_slice/stack:output:0<sequential/rot_equiv_conv2d_2/strided_slice/stack_1:output:0<sequential/rot_equiv_conv2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_conv2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RГ
!sequential/rot_equiv_conv2d_2/subSub4sequential/rot_equiv_conv2d_2/strided_slice:output:0,sequential/rot_equiv_conv2d_2/sub/y:output:0*
T0	*
_output_shapes
: ┤
3sequential/rot_equiv_conv2d_2/clip_by_value/MinimumMinimum,sequential/rot_equiv_conv2d_2/Const:output:0%sequential/rot_equiv_conv2d_2/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_conv2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╚
+sequential/rot_equiv_conv2d_2/clip_by_valueMaximum7sequential/rot_equiv_conv2d_2/clip_by_value/Minimum:z:06sequential/rot_equiv_conv2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_conv2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ц
&sequential/rot_equiv_conv2d_2/GatherV2GatherV2,sequential/rot_equiv_pool2d_1/stack:output:0/sequential/rot_equiv_conv2d_2/clip_by_value:z:04sequential/rot_equiv_conv2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" ┬
8sequential/rot_equiv_conv2d_2/convolution/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Ѕ
)sequential/rot_equiv_conv2d_2/convolutionConv2D/sequential/rot_equiv_conv2d_2/GatherV2:output:0@sequential/rot_equiv_conv2d_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RЉ
%sequential/rot_equiv_conv2d_2/Shape_1Shape,sequential/rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_2/strided_slice_1StridedSlice.sequential/rot_equiv_conv2d_2/Shape_1:output:0<sequential/rot_equiv_conv2d_2/strided_slice_1/stack:output:0>sequential/rot_equiv_conv2d_2/strided_slice_1/stack_1:output:0>sequential/rot_equiv_conv2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_2/sub_1Sub6sequential/rot_equiv_conv2d_2/strided_slice_1:output:0.sequential/rot_equiv_conv2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_2/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_conv2d_2/Const_1:output:0'sequential/rot_equiv_conv2d_2/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_2/clip_by_value_1Maximum9sequential/rot_equiv_conv2d_2/clip_by_value_1/Minimum:z:08sequential/rot_equiv_conv2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ф
(sequential/rot_equiv_conv2d_2/GatherV2_1GatherV2,sequential/rot_equiv_pool2d_1/stack:output:01sequential/rot_equiv_conv2d_2/clip_by_value_1:z:06sequential/rot_equiv_conv2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" ─
:sequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Ј
+sequential/rot_equiv_conv2d_2/convolution_1Conv2D1sequential/rot_equiv_conv2d_2/GatherV2_1:output:0Bsequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RЉ
%sequential/rot_equiv_conv2d_2/Shape_2Shape,sequential/rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_2/strided_slice_2StridedSlice.sequential/rot_equiv_conv2d_2/Shape_2:output:0<sequential/rot_equiv_conv2d_2/strided_slice_2/stack:output:0>sequential/rot_equiv_conv2d_2/strided_slice_2/stack_1:output:0>sequential/rot_equiv_conv2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_2/sub_2Sub6sequential/rot_equiv_conv2d_2/strided_slice_2:output:0.sequential/rot_equiv_conv2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_2/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_conv2d_2/Const_2:output:0'sequential/rot_equiv_conv2d_2/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_2/clip_by_value_2Maximum9sequential/rot_equiv_conv2d_2/clip_by_value_2/Minimum:z:08sequential/rot_equiv_conv2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ф
(sequential/rot_equiv_conv2d_2/GatherV2_2GatherV2,sequential/rot_equiv_pool2d_1/stack:output:01sequential/rot_equiv_conv2d_2/clip_by_value_2:z:06sequential/rot_equiv_conv2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" ─
:sequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Ј
+sequential/rot_equiv_conv2d_2/convolution_2Conv2D1sequential/rot_equiv_conv2d_2/GatherV2_2:output:0Bsequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RЉ
%sequential/rot_equiv_conv2d_2/Shape_3Shape,sequential/rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_2/strided_slice_3StridedSlice.sequential/rot_equiv_conv2d_2/Shape_3:output:0<sequential/rot_equiv_conv2d_2/strided_slice_3/stack:output:0>sequential/rot_equiv_conv2d_2/strided_slice_3/stack_1:output:0>sequential/rot_equiv_conv2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_2/sub_3Sub6sequential/rot_equiv_conv2d_2/strided_slice_3:output:0.sequential/rot_equiv_conv2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_2/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_conv2d_2/Const_3:output:0'sequential/rot_equiv_conv2d_2/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_2/clip_by_value_3Maximum9sequential/rot_equiv_conv2d_2/clip_by_value_3/Minimum:z:08sequential/rot_equiv_conv2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ф
(sequential/rot_equiv_conv2d_2/GatherV2_3GatherV2,sequential/rot_equiv_pool2d_1/stack:output:01sequential/rot_equiv_conv2d_2/clip_by_value_3:z:06sequential/rot_equiv_conv2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" ─
:sequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Ј
+sequential/rot_equiv_conv2d_2/convolution_3Conv2D1sequential/rot_equiv_conv2d_2/GatherV2_3:output:0Bsequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
П
#sequential/rot_equiv_conv2d_2/stackPack2sequential/rot_equiv_conv2d_2/convolution:output:04sequential/rot_equiv_conv2d_2/convolution_1:output:04sequential/rot_equiv_conv2d_2/convolution_2:output:04sequential/rot_equiv_conv2d_2/convolution_3:output:0*
N*
T0*3
_output_shapes!
:           @*
axis■        ќ
"sequential/rot_equiv_conv2d_2/ReluRelu,sequential/rot_equiv_conv2d_2/stack:output:0*
T0*3
_output_shapes!
:           @«
4sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp=sequential_rot_equiv_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0я
%sequential/rot_equiv_conv2d_2/BiasAddBiasAdd0sequential/rot_equiv_conv2d_2/Relu:activations:0<sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:           @e
#sequential/rot_equiv_pool2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Љ
#sequential/rot_equiv_pool2d_2/ShapeShape.sequential/rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	ё
1sequential/rot_equiv_pool2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        є
3sequential/rot_equiv_pool2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         }
3sequential/rot_equiv_pool2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
+sequential/rot_equiv_pool2d_2/strided_sliceStridedSlice,sequential/rot_equiv_pool2d_2/Shape:output:0:sequential/rot_equiv_pool2d_2/strided_slice/stack:output:0<sequential/rot_equiv_pool2d_2/strided_slice/stack_1:output:0<sequential/rot_equiv_pool2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RГ
!sequential/rot_equiv_pool2d_2/subSub4sequential/rot_equiv_pool2d_2/strided_slice:output:0,sequential/rot_equiv_pool2d_2/sub/y:output:0*
T0	*
_output_shapes
: ┤
3sequential/rot_equiv_pool2d_2/clip_by_value/MinimumMinimum,sequential/rot_equiv_pool2d_2/Const:output:0%sequential/rot_equiv_pool2d_2/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╚
+sequential/rot_equiv_pool2d_2/clip_by_valueMaximum7sequential/rot_equiv_pool2d_2/clip_by_value/Minimum:z:06sequential/rot_equiv_pool2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        д
&sequential/rot_equiv_pool2d_2/GatherV2GatherV2.sequential/rot_equiv_conv2d_2/BiasAdd:output:0/sequential/rot_equiv_pool2d_2/clip_by_value:z:04sequential/rot_equiv_pool2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @я
5sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPoolMaxPool/sequential/rot_equiv_pool2d_2/GatherV2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RЊ
%sequential/rot_equiv_pool2d_2/Shape_1Shape.sequential/rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_pool2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_pool2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_pool2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_pool2d_2/strided_slice_1StridedSlice.sequential/rot_equiv_pool2d_2/Shape_1:output:0<sequential/rot_equiv_pool2d_2/strided_slice_1/stack:output:0>sequential/rot_equiv_pool2d_2/strided_slice_1/stack_1:output:0>sequential/rot_equiv_pool2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_pool2d_2/sub_1Sub6sequential/rot_equiv_pool2d_2/strided_slice_1:output:0.sequential/rot_equiv_pool2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_pool2d_2/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_pool2d_2/Const_1:output:0'sequential/rot_equiv_pool2d_2/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_pool2d_2/clip_by_value_1Maximum9sequential/rot_equiv_pool2d_2/clip_by_value_1/Minimum:z:08sequential/rot_equiv_pool2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        г
(sequential/rot_equiv_pool2d_2/GatherV2_1GatherV2.sequential/rot_equiv_conv2d_2/BiasAdd:output:01sequential/rot_equiv_pool2d_2/clip_by_value_1:z:06sequential/rot_equiv_pool2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @Р
7sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1MaxPool1sequential/rot_equiv_pool2d_2/GatherV2_1:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RЊ
%sequential/rot_equiv_pool2d_2/Shape_2Shape.sequential/rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_pool2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_pool2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_pool2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_pool2d_2/strided_slice_2StridedSlice.sequential/rot_equiv_pool2d_2/Shape_2:output:0<sequential/rot_equiv_pool2d_2/strided_slice_2/stack:output:0>sequential/rot_equiv_pool2d_2/strided_slice_2/stack_1:output:0>sequential/rot_equiv_pool2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_pool2d_2/sub_2Sub6sequential/rot_equiv_pool2d_2/strided_slice_2:output:0.sequential/rot_equiv_pool2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_pool2d_2/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_pool2d_2/Const_2:output:0'sequential/rot_equiv_pool2d_2/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_pool2d_2/clip_by_value_2Maximum9sequential/rot_equiv_pool2d_2/clip_by_value_2/Minimum:z:08sequential/rot_equiv_pool2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        г
(sequential/rot_equiv_pool2d_2/GatherV2_2GatherV2.sequential/rot_equiv_conv2d_2/BiasAdd:output:01sequential/rot_equiv_pool2d_2/clip_by_value_2:z:06sequential/rot_equiv_pool2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @Р
7sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2MaxPool1sequential/rot_equiv_pool2d_2/GatherV2_2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RЊ
%sequential/rot_equiv_pool2d_2/Shape_3Shape.sequential/rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_pool2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_pool2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_pool2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_pool2d_2/strided_slice_3StridedSlice.sequential/rot_equiv_pool2d_2/Shape_3:output:0<sequential/rot_equiv_pool2d_2/strided_slice_3/stack:output:0>sequential/rot_equiv_pool2d_2/strided_slice_3/stack_1:output:0>sequential/rot_equiv_pool2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_pool2d_2/sub_3Sub6sequential/rot_equiv_pool2d_2/strided_slice_3:output:0.sequential/rot_equiv_pool2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_pool2d_2/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_pool2d_2/Const_3:output:0'sequential/rot_equiv_pool2d_2/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_pool2d_2/clip_by_value_3Maximum9sequential/rot_equiv_pool2d_2/clip_by_value_3/Minimum:z:08sequential/rot_equiv_pool2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        г
(sequential/rot_equiv_pool2d_2/GatherV2_3GatherV2.sequential/rot_equiv_conv2d_2/BiasAdd:output:01sequential/rot_equiv_pool2d_2/clip_by_value_3:z:06sequential/rot_equiv_pool2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @Р
7sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3MaxPool1sequential/rot_equiv_pool2d_2/GatherV2_3:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Ї
#sequential/rot_equiv_pool2d_2/stackPack>sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool:output:0@sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1:output:0@sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2:output:0@sequential/rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        e
#sequential/rot_equiv_conv2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ј
#sequential/rot_equiv_conv2d_3/ShapeShape,sequential/rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	ё
1sequential/rot_equiv_conv2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        є
3sequential/rot_equiv_conv2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         }
3sequential/rot_equiv_conv2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
+sequential/rot_equiv_conv2d_3/strided_sliceStridedSlice,sequential/rot_equiv_conv2d_3/Shape:output:0:sequential/rot_equiv_conv2d_3/strided_slice/stack:output:0<sequential/rot_equiv_conv2d_3/strided_slice/stack_1:output:0<sequential/rot_equiv_conv2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_conv2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RГ
!sequential/rot_equiv_conv2d_3/subSub4sequential/rot_equiv_conv2d_3/strided_slice:output:0,sequential/rot_equiv_conv2d_3/sub/y:output:0*
T0	*
_output_shapes
: ┤
3sequential/rot_equiv_conv2d_3/clip_by_value/MinimumMinimum,sequential/rot_equiv_conv2d_3/Const:output:0%sequential/rot_equiv_conv2d_3/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_conv2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╚
+sequential/rot_equiv_conv2d_3/clip_by_valueMaximum7sequential/rot_equiv_conv2d_3/clip_by_value/Minimum:z:06sequential/rot_equiv_conv2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_conv2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ц
&sequential/rot_equiv_conv2d_3/GatherV2GatherV2,sequential/rot_equiv_pool2d_2/stack:output:0/sequential/rot_equiv_conv2d_3/clip_by_value:z:04sequential/rot_equiv_conv2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @┬
8sequential/rot_equiv_conv2d_3/convolution/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ѕ
)sequential/rot_equiv_conv2d_3/convolutionConv2D/sequential/rot_equiv_conv2d_3/GatherV2:output:0@sequential/rot_equiv_conv2d_3/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RЉ
%sequential/rot_equiv_conv2d_3/Shape_1Shape,sequential/rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_3/strided_slice_1StridedSlice.sequential/rot_equiv_conv2d_3/Shape_1:output:0<sequential/rot_equiv_conv2d_3/strided_slice_1/stack:output:0>sequential/rot_equiv_conv2d_3/strided_slice_1/stack_1:output:0>sequential/rot_equiv_conv2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_3/sub_1Sub6sequential/rot_equiv_conv2d_3/strided_slice_1:output:0.sequential/rot_equiv_conv2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_3/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_conv2d_3/Const_1:output:0'sequential/rot_equiv_conv2d_3/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_3/clip_by_value_1Maximum9sequential/rot_equiv_conv2d_3/clip_by_value_1/Minimum:z:08sequential/rot_equiv_conv2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ф
(sequential/rot_equiv_conv2d_3/GatherV2_1GatherV2,sequential/rot_equiv_pool2d_2/stack:output:01sequential/rot_equiv_conv2d_3/clip_by_value_1:z:06sequential/rot_equiv_conv2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @─
:sequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ј
+sequential/rot_equiv_conv2d_3/convolution_1Conv2D1sequential/rot_equiv_conv2d_3/GatherV2_1:output:0Bsequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RЉ
%sequential/rot_equiv_conv2d_3/Shape_2Shape,sequential/rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_3/strided_slice_2StridedSlice.sequential/rot_equiv_conv2d_3/Shape_2:output:0<sequential/rot_equiv_conv2d_3/strided_slice_2/stack:output:0>sequential/rot_equiv_conv2d_3/strided_slice_2/stack_1:output:0>sequential/rot_equiv_conv2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_3/sub_2Sub6sequential/rot_equiv_conv2d_3/strided_slice_2:output:0.sequential/rot_equiv_conv2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_3/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_conv2d_3/Const_2:output:0'sequential/rot_equiv_conv2d_3/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_3/clip_by_value_2Maximum9sequential/rot_equiv_conv2d_3/clip_by_value_2/Minimum:z:08sequential/rot_equiv_conv2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ф
(sequential/rot_equiv_conv2d_3/GatherV2_2GatherV2,sequential/rot_equiv_pool2d_2/stack:output:01sequential/rot_equiv_conv2d_3/clip_by_value_2:z:06sequential/rot_equiv_conv2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @─
:sequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ј
+sequential/rot_equiv_conv2d_3/convolution_2Conv2D1sequential/rot_equiv_conv2d_3/GatherV2_2:output:0Bsequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RЉ
%sequential/rot_equiv_conv2d_3/Shape_3Shape,sequential/rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_3/strided_slice_3StridedSlice.sequential/rot_equiv_conv2d_3/Shape_3:output:0<sequential/rot_equiv_conv2d_3/strided_slice_3/stack:output:0>sequential/rot_equiv_conv2d_3/strided_slice_3/stack_1:output:0>sequential/rot_equiv_conv2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_3/sub_3Sub6sequential/rot_equiv_conv2d_3/strided_slice_3:output:0.sequential/rot_equiv_conv2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_3/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_conv2d_3/Const_3:output:0'sequential/rot_equiv_conv2d_3/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_3/clip_by_value_3Maximum9sequential/rot_equiv_conv2d_3/clip_by_value_3/Minimum:z:08sequential/rot_equiv_conv2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ф
(sequential/rot_equiv_conv2d_3/GatherV2_3GatherV2,sequential/rot_equiv_pool2d_2/stack:output:01sequential/rot_equiv_conv2d_3/clip_by_value_3:z:06sequential/rot_equiv_conv2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @─
:sequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ј
+sequential/rot_equiv_conv2d_3/convolution_3Conv2D1sequential/rot_equiv_conv2d_3/GatherV2_3:output:0Bsequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
П
#sequential/rot_equiv_conv2d_3/stackPack2sequential/rot_equiv_conv2d_3/convolution:output:04sequential/rot_equiv_conv2d_3/convolution_1:output:04sequential/rot_equiv_conv2d_3/convolution_2:output:04sequential/rot_equiv_conv2d_3/convolution_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        ќ
"sequential/rot_equiv_conv2d_3/ReluRelu,sequential/rot_equiv_conv2d_3/stack:output:0*
T0*3
_output_shapes!
:         @«
4sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp=sequential_rot_equiv_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0я
%sequential/rot_equiv_conv2d_3/BiasAddBiasAdd0sequential/rot_equiv_conv2d_3/Relu:activations:0<sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         @e
#sequential/rot_equiv_pool2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Љ
#sequential/rot_equiv_pool2d_3/ShapeShape.sequential/rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	ё
1sequential/rot_equiv_pool2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        є
3sequential/rot_equiv_pool2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         }
3sequential/rot_equiv_pool2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
+sequential/rot_equiv_pool2d_3/strided_sliceStridedSlice,sequential/rot_equiv_pool2d_3/Shape:output:0:sequential/rot_equiv_pool2d_3/strided_slice/stack:output:0<sequential/rot_equiv_pool2d_3/strided_slice/stack_1:output:0<sequential/rot_equiv_pool2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_pool2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RГ
!sequential/rot_equiv_pool2d_3/subSub4sequential/rot_equiv_pool2d_3/strided_slice:output:0,sequential/rot_equiv_pool2d_3/sub/y:output:0*
T0	*
_output_shapes
: ┤
3sequential/rot_equiv_pool2d_3/clip_by_value/MinimumMinimum,sequential/rot_equiv_pool2d_3/Const:output:0%sequential/rot_equiv_pool2d_3/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_pool2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╚
+sequential/rot_equiv_pool2d_3/clip_by_valueMaximum7sequential/rot_equiv_pool2d_3/clip_by_value/Minimum:z:06sequential/rot_equiv_pool2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_pool2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        д
&sequential/rot_equiv_pool2d_3/GatherV2GatherV2.sequential/rot_equiv_conv2d_3/BiasAdd:output:0/sequential/rot_equiv_pool2d_3/clip_by_value:z:04sequential/rot_equiv_pool2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @я
5sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPoolMaxPool/sequential/rot_equiv_pool2d_3/GatherV2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RЊ
%sequential/rot_equiv_pool2d_3/Shape_1Shape.sequential/rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_pool2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_pool2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_pool2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_pool2d_3/strided_slice_1StridedSlice.sequential/rot_equiv_pool2d_3/Shape_1:output:0<sequential/rot_equiv_pool2d_3/strided_slice_1/stack:output:0>sequential/rot_equiv_pool2d_3/strided_slice_1/stack_1:output:0>sequential/rot_equiv_pool2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_pool2d_3/sub_1Sub6sequential/rot_equiv_pool2d_3/strided_slice_1:output:0.sequential/rot_equiv_pool2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_pool2d_3/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_pool2d_3/Const_1:output:0'sequential/rot_equiv_pool2d_3/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_pool2d_3/clip_by_value_1Maximum9sequential/rot_equiv_pool2d_3/clip_by_value_1/Minimum:z:08sequential/rot_equiv_pool2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        г
(sequential/rot_equiv_pool2d_3/GatherV2_1GatherV2.sequential/rot_equiv_conv2d_3/BiasAdd:output:01sequential/rot_equiv_pool2d_3/clip_by_value_1:z:06sequential/rot_equiv_pool2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Р
7sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1MaxPool1sequential/rot_equiv_pool2d_3/GatherV2_1:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RЊ
%sequential/rot_equiv_pool2d_3/Shape_2Shape.sequential/rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_pool2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_pool2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_pool2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_pool2d_3/strided_slice_2StridedSlice.sequential/rot_equiv_pool2d_3/Shape_2:output:0<sequential/rot_equiv_pool2d_3/strided_slice_2/stack:output:0>sequential/rot_equiv_pool2d_3/strided_slice_2/stack_1:output:0>sequential/rot_equiv_pool2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_pool2d_3/sub_2Sub6sequential/rot_equiv_pool2d_3/strided_slice_2:output:0.sequential/rot_equiv_pool2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_pool2d_3/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_pool2d_3/Const_2:output:0'sequential/rot_equiv_pool2d_3/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_pool2d_3/clip_by_value_2Maximum9sequential/rot_equiv_pool2d_3/clip_by_value_2/Minimum:z:08sequential/rot_equiv_pool2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        г
(sequential/rot_equiv_pool2d_3/GatherV2_2GatherV2.sequential/rot_equiv_conv2d_3/BiasAdd:output:01sequential/rot_equiv_pool2d_3/clip_by_value_2:z:06sequential/rot_equiv_pool2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Р
7sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2MaxPool1sequential/rot_equiv_pool2d_3/GatherV2_2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
g
%sequential/rot_equiv_pool2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RЊ
%sequential/rot_equiv_pool2d_3/Shape_3Shape.sequential/rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_pool2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_pool2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_pool2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_pool2d_3/strided_slice_3StridedSlice.sequential/rot_equiv_pool2d_3/Shape_3:output:0<sequential/rot_equiv_pool2d_3/strided_slice_3/stack:output:0>sequential/rot_equiv_pool2d_3/strided_slice_3/stack_1:output:0>sequential/rot_equiv_pool2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_pool2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_pool2d_3/sub_3Sub6sequential/rot_equiv_pool2d_3/strided_slice_3:output:0.sequential/rot_equiv_pool2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_pool2d_3/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_pool2d_3/Const_3:output:0'sequential/rot_equiv_pool2d_3/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_pool2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_pool2d_3/clip_by_value_3Maximum9sequential/rot_equiv_pool2d_3/clip_by_value_3/Minimum:z:08sequential/rot_equiv_pool2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_pool2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        г
(sequential/rot_equiv_pool2d_3/GatherV2_3GatherV2.sequential/rot_equiv_conv2d_3/BiasAdd:output:01sequential/rot_equiv_pool2d_3/clip_by_value_3:z:06sequential/rot_equiv_pool2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Р
7sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3MaxPool1sequential/rot_equiv_pool2d_3/GatherV2_3:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Ї
#sequential/rot_equiv_pool2d_3/stackPack>sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool:output:0@sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1:output:0@sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2:output:0@sequential/rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        e
#sequential/rot_equiv_conv2d_4/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ј
#sequential/rot_equiv_conv2d_4/ShapeShape,sequential/rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	ё
1sequential/rot_equiv_conv2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        є
3sequential/rot_equiv_conv2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         }
3sequential/rot_equiv_conv2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
+sequential/rot_equiv_conv2d_4/strided_sliceStridedSlice,sequential/rot_equiv_conv2d_4/Shape:output:0:sequential/rot_equiv_conv2d_4/strided_slice/stack:output:0<sequential/rot_equiv_conv2d_4/strided_slice/stack_1:output:0<sequential/rot_equiv_conv2d_4/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#sequential/rot_equiv_conv2d_4/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RГ
!sequential/rot_equiv_conv2d_4/subSub4sequential/rot_equiv_conv2d_4/strided_slice:output:0,sequential/rot_equiv_conv2d_4/sub/y:output:0*
T0	*
_output_shapes
: ┤
3sequential/rot_equiv_conv2d_4/clip_by_value/MinimumMinimum,sequential/rot_equiv_conv2d_4/Const:output:0%sequential/rot_equiv_conv2d_4/sub:z:0*
T0	*
_output_shapes
: o
-sequential/rot_equiv_conv2d_4/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╚
+sequential/rot_equiv_conv2d_4/clip_by_valueMaximum7sequential/rot_equiv_conv2d_4/clip_by_value/Minimum:z:06sequential/rot_equiv_conv2d_4/clip_by_value/y:output:0*
T0	*
_output_shapes
: v
+sequential/rot_equiv_conv2d_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ц
&sequential/rot_equiv_conv2d_4/GatherV2GatherV2,sequential/rot_equiv_pool2d_3/stack:output:0/sequential/rot_equiv_conv2d_4/clip_by_value:z:04sequential/rot_equiv_conv2d_4/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @├
8sequential/rot_equiv_conv2d_4/convolution/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0і
)sequential/rot_equiv_conv2d_4/convolutionConv2D/sequential/rot_equiv_conv2d_4/GatherV2:output:0@sequential/rot_equiv_conv2d_4/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_4/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 RЉ
%sequential/rot_equiv_conv2d_4/Shape_1Shape,sequential/rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_4/strided_slice_1StridedSlice.sequential/rot_equiv_conv2d_4/Shape_1:output:0<sequential/rot_equiv_conv2d_4/strided_slice_1/stack:output:0>sequential/rot_equiv_conv2d_4/strided_slice_1/stack_1:output:0>sequential/rot_equiv_conv2d_4/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_4/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_4/sub_1Sub6sequential/rot_equiv_conv2d_4/strided_slice_1:output:0.sequential/rot_equiv_conv2d_4/sub_1/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_4/clip_by_value_1/MinimumMinimum.sequential/rot_equiv_conv2d_4/Const_1:output:0'sequential/rot_equiv_conv2d_4/sub_1:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_4/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_4/clip_by_value_1Maximum9sequential/rot_equiv_conv2d_4/clip_by_value_1/Minimum:z:08sequential/rot_equiv_conv2d_4/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ф
(sequential/rot_equiv_conv2d_4/GatherV2_1GatherV2,sequential/rot_equiv_pool2d_3/stack:output:01sequential/rot_equiv_conv2d_4/clip_by_value_1:z:06sequential/rot_equiv_conv2d_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @┼
:sequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0љ
+sequential/rot_equiv_conv2d_4/convolution_1Conv2D1sequential/rot_equiv_conv2d_4/GatherV2_1:output:0Bsequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_4/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 RЉ
%sequential/rot_equiv_conv2d_4/Shape_2Shape,sequential/rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_4/strided_slice_2StridedSlice.sequential/rot_equiv_conv2d_4/Shape_2:output:0<sequential/rot_equiv_conv2d_4/strided_slice_2/stack:output:0>sequential/rot_equiv_conv2d_4/strided_slice_2/stack_1:output:0>sequential/rot_equiv_conv2d_4/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_4/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_4/sub_2Sub6sequential/rot_equiv_conv2d_4/strided_slice_2:output:0.sequential/rot_equiv_conv2d_4/sub_2/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_4/clip_by_value_2/MinimumMinimum.sequential/rot_equiv_conv2d_4/Const_2:output:0'sequential/rot_equiv_conv2d_4/sub_2:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_4/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_4/clip_by_value_2Maximum9sequential/rot_equiv_conv2d_4/clip_by_value_2/Minimum:z:08sequential/rot_equiv_conv2d_4/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_4/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ф
(sequential/rot_equiv_conv2d_4/GatherV2_2GatherV2,sequential/rot_equiv_pool2d_3/stack:output:01sequential/rot_equiv_conv2d_4/clip_by_value_2:z:06sequential/rot_equiv_conv2d_4/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @┼
:sequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0љ
+sequential/rot_equiv_conv2d_4/convolution_2Conv2D1sequential/rot_equiv_conv2d_4/GatherV2_2:output:0Bsequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
g
%sequential/rot_equiv_conv2d_4/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 RЉ
%sequential/rot_equiv_conv2d_4/Shape_3Shape,sequential/rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	є
3sequential/rot_equiv_conv2d_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        ѕ
5sequential/rot_equiv_conv2d_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         
5sequential/rot_equiv_conv2d_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
-sequential/rot_equiv_conv2d_4/strided_slice_3StridedSlice.sequential/rot_equiv_conv2d_4/Shape_3:output:0<sequential/rot_equiv_conv2d_4/strided_slice_3/stack:output:0>sequential/rot_equiv_conv2d_4/strided_slice_3/stack_1:output:0>sequential/rot_equiv_conv2d_4/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%sequential/rot_equiv_conv2d_4/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R│
#sequential/rot_equiv_conv2d_4/sub_3Sub6sequential/rot_equiv_conv2d_4/strided_slice_3:output:0.sequential/rot_equiv_conv2d_4/sub_3/y:output:0*
T0	*
_output_shapes
: ║
5sequential/rot_equiv_conv2d_4/clip_by_value_3/MinimumMinimum.sequential/rot_equiv_conv2d_4/Const_3:output:0'sequential/rot_equiv_conv2d_4/sub_3:z:0*
T0	*
_output_shapes
: q
/sequential/rot_equiv_conv2d_4/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ╬
-sequential/rot_equiv_conv2d_4/clip_by_value_3Maximum9sequential/rot_equiv_conv2d_4/clip_by_value_3/Minimum:z:08sequential/rot_equiv_conv2d_4/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: x
-sequential/rot_equiv_conv2d_4/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ф
(sequential/rot_equiv_conv2d_4/GatherV2_3GatherV2,sequential/rot_equiv_pool2d_3/stack:output:01sequential/rot_equiv_conv2d_4/clip_by_value_3:z:06sequential/rot_equiv_conv2d_4/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @┼
:sequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOpReadVariableOpAsequential_rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0љ
+sequential/rot_equiv_conv2d_4/convolution_3Conv2D1sequential/rot_equiv_conv2d_4/GatherV2_3:output:0Bsequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
я
#sequential/rot_equiv_conv2d_4/stackPack2sequential/rot_equiv_conv2d_4/convolution:output:04sequential/rot_equiv_conv2d_4/convolution_1:output:04sequential/rot_equiv_conv2d_4/convolution_2:output:04sequential/rot_equiv_conv2d_4/convolution_3:output:0*
N*
T0*4
_output_shapes"
 :         ђ*
axis■        Ќ
"sequential/rot_equiv_conv2d_4/ReluRelu,sequential/rot_equiv_conv2d_4/stack:output:0*
T0*4
_output_shapes"
 :         ђ»
4sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp=sequential_rot_equiv_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0▀
%sequential/rot_equiv_conv2d_4/BiasAddBiasAdd0sequential/rot_equiv_conv2d_4/Relu:activations:0<sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         ђx
-sequential/rot_inv_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        ┼
sequential/rot_inv_pool/MaxMax.sequential/rot_equiv_conv2d_4/BiasAdd:output:06sequential/rot_inv_pool/Max/reduction_indices:output:0*
T0*0
_output_shapes
:         ђi
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  А
sequential/flatten/ReshapeReshape$sequential/rot_inv_pool/Max:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:         ђЌ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype0е
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Е
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:          џ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0г
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ў
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Ї
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp3^sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOp+^sequential/rot_equiv_conv2d/ReadVariableOp-^sequential/rot_equiv_conv2d/ReadVariableOp_1-^sequential/rot_equiv_conv2d/ReadVariableOp_27^sequential/rot_equiv_conv2d/convolution/ReadVariableOp5^sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOp9^sequential/rot_equiv_conv2d_1/convolution/ReadVariableOp;^sequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOp;^sequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOp;^sequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOp5^sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOp9^sequential/rot_equiv_conv2d_2/convolution/ReadVariableOp;^sequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOp;^sequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOp;^sequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOp5^sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOp9^sequential/rot_equiv_conv2d_3/convolution/ReadVariableOp;^sequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOp;^sequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOp;^sequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOp5^sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOp9^sequential/rot_equiv_conv2d_4/convolution/ReadVariableOp;^sequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOp;^sequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOp;^sequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2h
2sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOp2sequential/rot_equiv_conv2d/BiasAdd/ReadVariableOp2X
*sequential/rot_equiv_conv2d/ReadVariableOp*sequential/rot_equiv_conv2d/ReadVariableOp2\
,sequential/rot_equiv_conv2d/ReadVariableOp_1,sequential/rot_equiv_conv2d/ReadVariableOp_12\
,sequential/rot_equiv_conv2d/ReadVariableOp_2,sequential/rot_equiv_conv2d/ReadVariableOp_22p
6sequential/rot_equiv_conv2d/convolution/ReadVariableOp6sequential/rot_equiv_conv2d/convolution/ReadVariableOp2l
4sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOp4sequential/rot_equiv_conv2d_1/BiasAdd/ReadVariableOp2t
8sequential/rot_equiv_conv2d_1/convolution/ReadVariableOp8sequential/rot_equiv_conv2d_1/convolution/ReadVariableOp2x
:sequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOp:sequential/rot_equiv_conv2d_1/convolution_1/ReadVariableOp2x
:sequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOp:sequential/rot_equiv_conv2d_1/convolution_2/ReadVariableOp2x
:sequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOp:sequential/rot_equiv_conv2d_1/convolution_3/ReadVariableOp2l
4sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOp4sequential/rot_equiv_conv2d_2/BiasAdd/ReadVariableOp2t
8sequential/rot_equiv_conv2d_2/convolution/ReadVariableOp8sequential/rot_equiv_conv2d_2/convolution/ReadVariableOp2x
:sequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOp:sequential/rot_equiv_conv2d_2/convolution_1/ReadVariableOp2x
:sequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOp:sequential/rot_equiv_conv2d_2/convolution_2/ReadVariableOp2x
:sequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOp:sequential/rot_equiv_conv2d_2/convolution_3/ReadVariableOp2l
4sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOp4sequential/rot_equiv_conv2d_3/BiasAdd/ReadVariableOp2t
8sequential/rot_equiv_conv2d_3/convolution/ReadVariableOp8sequential/rot_equiv_conv2d_3/convolution/ReadVariableOp2x
:sequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOp:sequential/rot_equiv_conv2d_3/convolution_1/ReadVariableOp2x
:sequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOp:sequential/rot_equiv_conv2d_3/convolution_2/ReadVariableOp2x
:sequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOp:sequential/rot_equiv_conv2d_3/convolution_3/ReadVariableOp2l
4sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOp4sequential/rot_equiv_conv2d_4/BiasAdd/ReadVariableOp2t
8sequential/rot_equiv_conv2d_4/convolution/ReadVariableOp8sequential/rot_equiv_conv2d_4/convolution/ReadVariableOp2x
:sequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOp:sequential/rot_equiv_conv2d_4/convolution_1/ReadVariableOp2x
:sequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOp:sequential/rot_equiv_conv2d_4/convolution_2/ReadVariableOp2x
:sequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOp:sequential/rot_equiv_conv2d_4/convolution_3/ReadVariableOp:i e
1
_output_shapes
:         љљ
0
_user_specified_namerot_equiv_conv2d_input
Њ
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_369903

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Љ
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_372997

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ы
d
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_372937

inputs
identity`
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:         ђ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ:\ X
4
_output_shapes"
 :         ђ
 
_user_specified_nameinputs
тC
Ь
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_370131

inputs=
#convolution_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбconvolution/ReadVariableOpбconvolution_1/ReadVariableOpбconvolution_2/ReadVariableOpбconvolution_3/ReadVariableOpG
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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG є
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0»
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG ѕ
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0х
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG ѕ
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0х
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG ѕ
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0х
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
К
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:         EE *
axis■        Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:         EE r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0ё
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         EE k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:         EE ┘
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         GG : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:         GG 
 
_user_specified_nameinputs
│У
«
F__inference_sequential_layer_call_and_return_conditional_losses_371670

inputsN
4rot_equiv_conv2d_convolution_readvariableop_resource: >
0rot_equiv_conv2d_biasadd_readvariableop_resource: P
6rot_equiv_conv2d_1_convolution_readvariableop_resource:  @
2rot_equiv_conv2d_1_biasadd_readvariableop_resource: P
6rot_equiv_conv2d_2_convolution_readvariableop_resource: @@
2rot_equiv_conv2d_2_biasadd_readvariableop_resource:@P
6rot_equiv_conv2d_3_convolution_readvariableop_resource:@@@
2rot_equiv_conv2d_3_biasadd_readvariableop_resource:@Q
6rot_equiv_conv2d_4_convolution_readvariableop_resource:@ђA
2rot_equiv_conv2d_4_biasadd_readvariableop_resource:	ђ7
$dense_matmul_readvariableop_resource:	ђ 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpб'rot_equiv_conv2d/BiasAdd/ReadVariableOpбrot_equiv_conv2d/ReadVariableOpб!rot_equiv_conv2d/ReadVariableOp_1б!rot_equiv_conv2d/ReadVariableOp_2б+rot_equiv_conv2d/convolution/ReadVariableOpб)rot_equiv_conv2d_1/BiasAdd/ReadVariableOpб-rot_equiv_conv2d_1/convolution/ReadVariableOpб/rot_equiv_conv2d_1/convolution_1/ReadVariableOpб/rot_equiv_conv2d_1/convolution_2/ReadVariableOpб/rot_equiv_conv2d_1/convolution_3/ReadVariableOpб)rot_equiv_conv2d_2/BiasAdd/ReadVariableOpб-rot_equiv_conv2d_2/convolution/ReadVariableOpб/rot_equiv_conv2d_2/convolution_1/ReadVariableOpб/rot_equiv_conv2d_2/convolution_2/ReadVariableOpб/rot_equiv_conv2d_2/convolution_3/ReadVariableOpб)rot_equiv_conv2d_3/BiasAdd/ReadVariableOpб-rot_equiv_conv2d_3/convolution/ReadVariableOpб/rot_equiv_conv2d_3/convolution_1/ReadVariableOpб/rot_equiv_conv2d_3/convolution_2/ReadVariableOpб/rot_equiv_conv2d_3/convolution_3/ReadVariableOpб)rot_equiv_conv2d_4/BiasAdd/ReadVariableOpб-rot_equiv_conv2d_4/convolution/ReadVariableOpб/rot_equiv_conv2d_4/convolution_1/ReadVariableOpб/rot_equiv_conv2d_4/convolution_2/ReadVariableOpб/rot_equiv_conv2d_4/convolution_3/ReadVariableOpе
+rot_equiv_conv2d/convolution/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0╚
rot_equiv_conv2d/convolutionConv2Dinputs3rot_equiv_conv2d/convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
А
$rot_equiv_conv2d/Rank/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0W
rot_equiv_conv2d/RankConst*
_output_shapes
: *
dtype0*
value	B :^
rot_equiv_conv2d/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
rot_equiv_conv2d/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Е
rot_equiv_conv2d/rangeRange%rot_equiv_conv2d/range/start:output:0rot_equiv_conv2d/Rank:output:0%rot_equiv_conv2d/range/delta:output:0*
_output_shapes
:Ё
,rot_equiv_conv2d/TensorScatterUpdate/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       }
,rot_equiv_conv2d/TensorScatterUpdate/updatesConst*
_output_shapes
:*
dtype0*
valueB"        
$rot_equiv_conv2d/TensorScatterUpdateTensorScatterUpdaterot_equiv_conv2d/range:output:05rot_equiv_conv2d/TensorScatterUpdate/indices:output:05rot_equiv_conv2d/TensorScatterUpdate/updates:output:0*
T0*
Tindices0*
_output_shapes
:ю
rot_equiv_conv2d/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :i
rot_equiv_conv2d/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:Ф
rot_equiv_conv2d/ReverseV2	ReverseV2'rot_equiv_conv2d/ReadVariableOp:value:0(rot_equiv_conv2d/ReverseV2/axis:output:0*
T0*&
_output_shapes
: г
rot_equiv_conv2d/transpose	Transpose#rot_equiv_conv2d/ReverseV2:output:0-rot_equiv_conv2d/TensorScatterUpdate:output:0*
T0*&
_output_shapes
: х
rot_equiv_conv2d/convolution_1Conv2Dinputsrot_equiv_conv2d/transpose:y:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
Y
rot_equiv_conv2d/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :▒
rot_equiv_conv2d/range_1Range'rot_equiv_conv2d/range_1/start:output:0 rot_equiv_conv2d/Rank_2:output:0'rot_equiv_conv2d/range_1/delta:output:0*
_output_shapes
:Є
.rot_equiv_conv2d/TensorScatterUpdate_1/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      
.rot_equiv_conv2d/TensorScatterUpdate_1/updatesConst*
_output_shapes
:*
dtype0*
valueB"      Є
&rot_equiv_conv2d/TensorScatterUpdate_1TensorScatterUpdate!rot_equiv_conv2d/range_1:output:07rot_equiv_conv2d/TensorScatterUpdate_1/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:┐
rot_equiv_conv2d/transpose_1	Transpose'rot_equiv_conv2d/convolution_1:output:0/rot_equiv_conv2d/TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:         јј Y
rot_equiv_conv2d/Rank_3Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:│
rot_equiv_conv2d/ReverseV2_1	ReverseV2 rot_equiv_conv2d/transpose_1:y:0*rot_equiv_conv2d/ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:         јј Б
&rot_equiv_conv2d/Rank_4/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_4Const*
_output_shapes
: *
dtype0*
value	B :ъ
!rot_equiv_conv2d/ReadVariableOp_1ReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Y
rot_equiv_conv2d/Rank_5Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_2/axisConst*
_output_shapes
:*
dtype0*
valueB: ▒
rot_equiv_conv2d/ReverseV2_2	ReverseV2)rot_equiv_conv2d/ReadVariableOp_1:value:0*rot_equiv_conv2d/ReverseV2_2/axis:output:0*
T0*&
_output_shapes
: Y
rot_equiv_conv2d/Rank_6Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_3/axisConst*
_output_shapes
:*
dtype0*
valueB:Г
rot_equiv_conv2d/ReverseV2_3	ReverseV2%rot_equiv_conv2d/ReverseV2_2:output:0*rot_equiv_conv2d/ReverseV2_3/axis:output:0*
T0*&
_output_shapes
: ╝
rot_equiv_conv2d/convolution_2Conv2Dinputs%rot_equiv_conv2d/ReverseV2_3:output:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
Y
rot_equiv_conv2d/Rank_7Const*
_output_shapes
: *
dtype0*
value	B :Y
rot_equiv_conv2d/Rank_8Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_4/axisConst*
_output_shapes
:*
dtype0*
valueB:║
rot_equiv_conv2d/ReverseV2_4	ReverseV2'rot_equiv_conv2d/convolution_2:output:0*rot_equiv_conv2d/ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:         јј Y
rot_equiv_conv2d/Rank_9Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:И
rot_equiv_conv2d/ReverseV2_5	ReverseV2%rot_equiv_conv2d/ReverseV2_4:output:0*rot_equiv_conv2d/ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:         јј ц
'rot_equiv_conv2d/Rank_10/ReadVariableOpReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Z
rot_equiv_conv2d/Rank_10Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :▓
rot_equiv_conv2d/range_2Range'rot_equiv_conv2d/range_2/start:output:0!rot_equiv_conv2d/Rank_10:output:0'rot_equiv_conv2d/range_2/delta:output:0*
_output_shapes
:Є
.rot_equiv_conv2d/TensorScatterUpdate_2/indicesConst*
_output_shapes

:*
dtype0*!
valueB"       
.rot_equiv_conv2d/TensorScatterUpdate_2/updatesConst*
_output_shapes
:*
dtype0*
valueB"       Є
&rot_equiv_conv2d/TensorScatterUpdate_2TensorScatterUpdate!rot_equiv_conv2d/range_2:output:07rot_equiv_conv2d/TensorScatterUpdate_2/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:ъ
!rot_equiv_conv2d/ReadVariableOp_2ReadVariableOp4rot_equiv_conv2d_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Х
rot_equiv_conv2d/transpose_2	Transpose)rot_equiv_conv2d/ReadVariableOp_2:value:0/rot_equiv_conv2d/TensorScatterUpdate_2:output:0*
T0*&
_output_shapes
: Z
rot_equiv_conv2d/Rank_11Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_6/axisConst*
_output_shapes
:*
dtype0*
valueB:е
rot_equiv_conv2d/ReverseV2_6	ReverseV2 rot_equiv_conv2d/transpose_2:y:0*rot_equiv_conv2d/ReverseV2_6/axis:output:0*
T0*&
_output_shapes
: ╝
rot_equiv_conv2d/convolution_3Conv2Dinputs%rot_equiv_conv2d/ReverseV2_6:output:0*
T0*1
_output_shapes
:         јј *
paddingVALID*
strides
Z
rot_equiv_conv2d/Rank_12Const*
_output_shapes
: *
dtype0*
value	B :`
rot_equiv_conv2d/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : `
rot_equiv_conv2d/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :▓
rot_equiv_conv2d/range_3Range'rot_equiv_conv2d/range_3/start:output:0!rot_equiv_conv2d/Rank_12:output:0'rot_equiv_conv2d/range_3/delta:output:0*
_output_shapes
:Є
.rot_equiv_conv2d/TensorScatterUpdate_3/indicesConst*
_output_shapes

:*
dtype0*!
valueB"      
.rot_equiv_conv2d/TensorScatterUpdate_3/updatesConst*
_output_shapes
:*
dtype0*
valueB"      Є
&rot_equiv_conv2d/TensorScatterUpdate_3TensorScatterUpdate!rot_equiv_conv2d/range_3:output:07rot_equiv_conv2d/TensorScatterUpdate_3/indices:output:07rot_equiv_conv2d/TensorScatterUpdate_3/updates:output:0*
T0*
Tindices0*
_output_shapes
:Z
rot_equiv_conv2d/Rank_13Const*
_output_shapes
: *
dtype0*
value	B :k
!rot_equiv_conv2d/ReverseV2_7/axisConst*
_output_shapes
:*
dtype0*
valueB:║
rot_equiv_conv2d/ReverseV2_7	ReverseV2'rot_equiv_conv2d/convolution_3:output:0*rot_equiv_conv2d/ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:         јј й
rot_equiv_conv2d/transpose_3	Transpose%rot_equiv_conv2d/ReverseV2_7:output:0/rot_equiv_conv2d/TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:         јј Њ
rot_equiv_conv2d/stackPack%rot_equiv_conv2d/convolution:output:0%rot_equiv_conv2d/ReverseV2_1:output:0%rot_equiv_conv2d/ReverseV2_5:output:0 rot_equiv_conv2d/transpose_3:y:0*
N*
T0*5
_output_shapes#
!:         јј *
axis■        ~
rot_equiv_conv2d/ReluRelurot_equiv_conv2d/stack:output:0*
T0*5
_output_shapes#
!:         јј ћ
'rot_equiv_conv2d/BiasAdd/ReadVariableOpReadVariableOp0rot_equiv_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╣
rot_equiv_conv2d/BiasAddBiasAdd#rot_equiv_conv2d/Relu:activations:0/rot_equiv_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:         јј X
rot_equiv_pool2d/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R w
rot_equiv_pool2d/ShapeShape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	w
$rot_equiv_pool2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        y
&rot_equiv_pool2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         p
&rot_equiv_pool2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
rot_equiv_pool2d/strided_sliceStridedSlicerot_equiv_pool2d/Shape:output:0-rot_equiv_pool2d/strided_slice/stack:output:0/rot_equiv_pool2d/strided_slice/stack_1:output:0/rot_equiv_pool2d/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskX
rot_equiv_pool2d/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rє
rot_equiv_pool2d/subSub'rot_equiv_pool2d/strided_slice:output:0rot_equiv_pool2d/sub/y:output:0*
T0	*
_output_shapes
: Ї
&rot_equiv_pool2d/clip_by_value/MinimumMinimumrot_equiv_pool2d/Const:output:0rot_equiv_pool2d/sub:z:0*
T0	*
_output_shapes
: b
 rot_equiv_pool2d/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R А
rot_equiv_pool2d/clip_by_valueMaximum*rot_equiv_pool2d/clip_by_value/Minimum:z:0)rot_equiv_pool2d/clip_by_value/y:output:0*
T0	*
_output_shapes
: i
rot_equiv_pool2d/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        З
rot_equiv_pool2d/GatherV2GatherV2!rot_equiv_conv2d/BiasAdd:output:0"rot_equiv_pool2d/clip_by_value:z:0'rot_equiv_pool2d/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ┬
&rot_equiv_pool2d/max_pooling2d/MaxPoolMaxPool"rot_equiv_pool2d/GatherV2:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_1Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d/strided_slice_1StridedSlice!rot_equiv_pool2d/Shape_1:output:0/rot_equiv_pool2d/strided_slice_1/stack:output:01rot_equiv_pool2d/strided_slice_1/stack_1:output:01rot_equiv_pool2d/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d/sub_1Sub)rot_equiv_pool2d/strided_slice_1:output:0!rot_equiv_pool2d/sub_1/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d/clip_by_value_1/MinimumMinimum!rot_equiv_pool2d/Const_1:output:0rot_equiv_pool2d/sub_1:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d/clip_by_value_1Maximum,rot_equiv_pool2d/clip_by_value_1/Minimum:z:0+rot_equiv_pool2d/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d/GatherV2_1GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_1:z:0)rot_equiv_pool2d/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј к
(rot_equiv_pool2d/max_pooling2d/MaxPool_1MaxPool$rot_equiv_pool2d/GatherV2_1:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_2Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d/strided_slice_2StridedSlice!rot_equiv_pool2d/Shape_2:output:0/rot_equiv_pool2d/strided_slice_2/stack:output:01rot_equiv_pool2d/strided_slice_2/stack_1:output:01rot_equiv_pool2d/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d/sub_2Sub)rot_equiv_pool2d/strided_slice_2:output:0!rot_equiv_pool2d/sub_2/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d/clip_by_value_2/MinimumMinimum!rot_equiv_pool2d/Const_2:output:0rot_equiv_pool2d/sub_2:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d/clip_by_value_2Maximum,rot_equiv_pool2d/clip_by_value_2/Minimum:z:0+rot_equiv_pool2d/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d/GatherV2_2GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_2:z:0)rot_equiv_pool2d/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј к
(rot_equiv_pool2d/max_pooling2d/MaxPool_2MaxPool$rot_equiv_pool2d/GatherV2_2:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
Z
rot_equiv_pool2d/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_pool2d/Shape_3Shape!rot_equiv_conv2d/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d/strided_slice_3StridedSlice!rot_equiv_pool2d/Shape_3:output:0/rot_equiv_pool2d/strided_slice_3/stack:output:01rot_equiv_pool2d/strided_slice_3/stack_1:output:01rot_equiv_pool2d/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d/sub_3Sub)rot_equiv_pool2d/strided_slice_3:output:0!rot_equiv_pool2d/sub_3/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d/clip_by_value_3/MinimumMinimum!rot_equiv_pool2d/Const_3:output:0rot_equiv_pool2d/sub_3:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d/clip_by_value_3Maximum,rot_equiv_pool2d/clip_by_value_3/Minimum:z:0+rot_equiv_pool2d/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d/GatherV2_3GatherV2!rot_equiv_conv2d/BiasAdd:output:0$rot_equiv_pool2d/clip_by_value_3:z:0)rot_equiv_pool2d/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј к
(rot_equiv_pool2d/max_pooling2d/MaxPool_3MaxPool$rot_equiv_pool2d/GatherV2_3:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
─
rot_equiv_pool2d/stackPack/rot_equiv_pool2d/max_pooling2d/MaxPool:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_1:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_2:output:01rot_equiv_pool2d/max_pooling2d/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         GG *
axis■        Z
rot_equiv_conv2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R w
rot_equiv_conv2d_1/ShapeShaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_conv2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_conv2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_conv2d_1/strided_sliceStridedSlice!rot_equiv_conv2d_1/Shape:output:0/rot_equiv_conv2d_1/strided_slice/stack:output:01rot_equiv_conv2d_1/strided_slice/stack_1:output:01rot_equiv_conv2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_conv2d_1/subSub)rot_equiv_conv2d_1/strided_slice:output:0!rot_equiv_conv2d_1/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_conv2d_1/clip_by_value/MinimumMinimum!rot_equiv_conv2d_1/Const:output:0rot_equiv_conv2d_1/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_conv2d_1/clip_by_valueMaximum,rot_equiv_conv2d_1/clip_by_value/Minimum:z:0+rot_equiv_conv2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Ш
rot_equiv_conv2d_1/GatherV2GatherV2rot_equiv_pool2d/stack:output:0$rot_equiv_conv2d_1/clip_by_value:z:0)rot_equiv_conv2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG г
-rot_equiv_conv2d_1/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0У
rot_equiv_conv2d_1/convolutionConv2D$rot_equiv_conv2d_1/GatherV2:output:05rot_equiv_conv2d_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_1Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_1/strided_slice_1StridedSlice#rot_equiv_conv2d_1/Shape_1:output:01rot_equiv_conv2d_1/strided_slice_1/stack:output:03rot_equiv_conv2d_1/strided_slice_1/stack_1:output:03rot_equiv_conv2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_1/sub_1Sub+rot_equiv_conv2d_1/strided_slice_1:output:0#rot_equiv_conv2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_1/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_1/Const_1:output:0rot_equiv_conv2d_1/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_1/clip_by_value_1Maximum.rot_equiv_conv2d_1/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Ч
rot_equiv_conv2d_1/GatherV2_1GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_1:z:0+rot_equiv_conv2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG «
/rot_equiv_conv2d_1/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Ь
 rot_equiv_conv2d_1/convolution_1Conv2D&rot_equiv_conv2d_1/GatherV2_1:output:07rot_equiv_conv2d_1/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_2Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_1/strided_slice_2StridedSlice#rot_equiv_conv2d_1/Shape_2:output:01rot_equiv_conv2d_1/strided_slice_2/stack:output:03rot_equiv_conv2d_1/strided_slice_2/stack_1:output:03rot_equiv_conv2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_1/sub_2Sub+rot_equiv_conv2d_1/strided_slice_2:output:0#rot_equiv_conv2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_1/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_1/Const_2:output:0rot_equiv_conv2d_1/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_1/clip_by_value_2Maximum.rot_equiv_conv2d_1/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Ч
rot_equiv_conv2d_1/GatherV2_2GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_2:z:0+rot_equiv_conv2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG «
/rot_equiv_conv2d_1/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Ь
 rot_equiv_conv2d_1/convolution_2Conv2D&rot_equiv_conv2d_1/GatherV2_2:output:07rot_equiv_conv2d_1/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
\
rot_equiv_conv2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 Ry
rot_equiv_conv2d_1/Shape_3Shaperot_equiv_pool2d/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_1/strided_slice_3StridedSlice#rot_equiv_conv2d_1/Shape_3:output:01rot_equiv_conv2d_1/strided_slice_3/stack:output:03rot_equiv_conv2d_1/strided_slice_3/stack_1:output:03rot_equiv_conv2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_1/sub_3Sub+rot_equiv_conv2d_1/strided_slice_3:output:0#rot_equiv_conv2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_1/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_1/Const_3:output:0rot_equiv_conv2d_1/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_1/clip_by_value_3Maximum.rot_equiv_conv2d_1/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Ч
rot_equiv_conv2d_1/GatherV2_3GatherV2rot_equiv_pool2d/stack:output:0&rot_equiv_conv2d_1/clip_by_value_3:z:0+rot_equiv_conv2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG «
/rot_equiv_conv2d_1/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0Ь
 rot_equiv_conv2d_1/convolution_3Conv2D&rot_equiv_conv2d_1/GatherV2_3:output:07rot_equiv_conv2d_1/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
д
rot_equiv_conv2d_1/stackPack'rot_equiv_conv2d_1/convolution:output:0)rot_equiv_conv2d_1/convolution_1:output:0)rot_equiv_conv2d_1/convolution_2:output:0)rot_equiv_conv2d_1/convolution_3:output:0*
N*
T0*3
_output_shapes!
:         EE *
axis■        ђ
rot_equiv_conv2d_1/ReluRelu!rot_equiv_conv2d_1/stack:output:0*
T0*3
_output_shapes!
:         EE ў
)rot_equiv_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0й
rot_equiv_conv2d_1/BiasAddBiasAdd%rot_equiv_conv2d_1/Relu:activations:01rot_equiv_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         EE Z
rot_equiv_pool2d_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_1/ShapeShape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d_1/strided_sliceStridedSlice!rot_equiv_pool2d_1/Shape:output:0/rot_equiv_pool2d_1/strided_slice/stack:output:01rot_equiv_pool2d_1/strided_slice/stack_1:output:01rot_equiv_pool2d_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d_1/subSub)rot_equiv_pool2d_1/strided_slice:output:0!rot_equiv_pool2d_1/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d_1/clip_by_value/MinimumMinimum!rot_equiv_pool2d_1/Const:output:0rot_equiv_pool2d_1/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_1/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d_1/clip_by_valueMaximum,rot_equiv_pool2d_1/clip_by_value/Minimum:z:0+rot_equiv_pool2d_1/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d_1/GatherV2GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0$rot_equiv_pool2d_1/clip_by_value:z:0)rot_equiv_pool2d_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE ╚
*rot_equiv_pool2d_1/max_pooling2d_1/MaxPoolMaxPool$rot_equiv_pool2d_1/GatherV2:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_1Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_1/strided_slice_1StridedSlice#rot_equiv_pool2d_1/Shape_1:output:01rot_equiv_pool2d_1/strided_slice_1/stack:output:03rot_equiv_pool2d_1/strided_slice_1/stack_1:output:03rot_equiv_pool2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_1/sub_1Sub+rot_equiv_pool2d_1/strided_slice_1:output:0#rot_equiv_pool2d_1/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_1/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_1/Const_1:output:0rot_equiv_pool2d_1/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_1/clip_by_value_1Maximum.rot_equiv_pool2d_1/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_1/GatherV2_1GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_1:z:0+rot_equiv_pool2d_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE ╠
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1MaxPool&rot_equiv_pool2d_1/GatherV2_1:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_2Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_1/strided_slice_2StridedSlice#rot_equiv_pool2d_1/Shape_2:output:01rot_equiv_pool2d_1/strided_slice_2/stack:output:03rot_equiv_pool2d_1/strided_slice_2/stack_1:output:03rot_equiv_pool2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_1/sub_2Sub+rot_equiv_pool2d_1/strided_slice_2:output:0#rot_equiv_pool2d_1/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_1/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_1/Const_2:output:0rot_equiv_pool2d_1/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_1/clip_by_value_2Maximum.rot_equiv_pool2d_1/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_1/GatherV2_2GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_2:z:0+rot_equiv_pool2d_1/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE ╠
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2MaxPool&rot_equiv_pool2d_1/GatherV2_2:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_1/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_1/Shape_3Shape#rot_equiv_conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_1/strided_slice_3StridedSlice#rot_equiv_pool2d_1/Shape_3:output:01rot_equiv_pool2d_1/strided_slice_3/stack:output:03rot_equiv_pool2d_1/strided_slice_3/stack_1:output:03rot_equiv_pool2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_1/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_1/sub_3Sub+rot_equiv_pool2d_1/strided_slice_3:output:0#rot_equiv_pool2d_1/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_1/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_1/Const_3:output:0rot_equiv_pool2d_1/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_1/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_1/clip_by_value_3Maximum.rot_equiv_pool2d_1/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_1/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_1/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_1/GatherV2_3GatherV2#rot_equiv_conv2d_1/BiasAdd:output:0&rot_equiv_pool2d_1/clip_by_value_3:z:0+rot_equiv_pool2d_1/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE ╠
,rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3MaxPool&rot_equiv_pool2d_1/GatherV2_3:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
о
rot_equiv_pool2d_1/stackPack3rot_equiv_pool2d_1/max_pooling2d_1/MaxPool:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_1:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_2:output:05rot_equiv_pool2d_1/max_pooling2d_1/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         "" *
axis■        Z
rot_equiv_conv2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_2/ShapeShape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_conv2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_conv2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_conv2d_2/strided_sliceStridedSlice!rot_equiv_conv2d_2/Shape:output:0/rot_equiv_conv2d_2/strided_slice/stack:output:01rot_equiv_conv2d_2/strided_slice/stack_1:output:01rot_equiv_conv2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_conv2d_2/subSub)rot_equiv_conv2d_2/strided_slice:output:0!rot_equiv_conv2d_2/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_conv2d_2/clip_by_value/MinimumMinimum!rot_equiv_conv2d_2/Const:output:0rot_equiv_conv2d_2/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_conv2d_2/clip_by_valueMaximum,rot_equiv_conv2d_2/clip_by_value/Minimum:z:0+rot_equiv_conv2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Э
rot_equiv_conv2d_2/GatherV2GatherV2!rot_equiv_pool2d_1/stack:output:0$rot_equiv_conv2d_2/clip_by_value:z:0)rot_equiv_conv2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" г
-rot_equiv_conv2d_2/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0У
rot_equiv_conv2d_2/convolutionConv2D$rot_equiv_conv2d_2/GatherV2:output:05rot_equiv_conv2d_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_1Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_2/strided_slice_1StridedSlice#rot_equiv_conv2d_2/Shape_1:output:01rot_equiv_conv2d_2/strided_slice_1/stack:output:03rot_equiv_conv2d_2/strided_slice_1/stack_1:output:03rot_equiv_conv2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_2/sub_1Sub+rot_equiv_conv2d_2/strided_slice_1:output:0#rot_equiv_conv2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_2/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_2/Const_1:output:0rot_equiv_conv2d_2/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_2/clip_by_value_1Maximum.rot_equiv_conv2d_2/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_2/GatherV2_1GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_1:z:0+rot_equiv_conv2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" «
/rot_equiv_conv2d_2/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Ь
 rot_equiv_conv2d_2/convolution_1Conv2D&rot_equiv_conv2d_2/GatherV2_1:output:07rot_equiv_conv2d_2/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_2Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_2/strided_slice_2StridedSlice#rot_equiv_conv2d_2/Shape_2:output:01rot_equiv_conv2d_2/strided_slice_2/stack:output:03rot_equiv_conv2d_2/strided_slice_2/stack_1:output:03rot_equiv_conv2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_2/sub_2Sub+rot_equiv_conv2d_2/strided_slice_2:output:0#rot_equiv_conv2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_2/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_2/Const_2:output:0rot_equiv_conv2d_2/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_2/clip_by_value_2Maximum.rot_equiv_conv2d_2/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_2/GatherV2_2GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_2:z:0+rot_equiv_conv2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" «
/rot_equiv_conv2d_2/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Ь
 rot_equiv_conv2d_2/convolution_2Conv2D&rot_equiv_conv2d_2/GatherV2_2:output:07rot_equiv_conv2d_2/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
\
rot_equiv_conv2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_2/Shape_3Shape!rot_equiv_pool2d_1/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_2/strided_slice_3StridedSlice#rot_equiv_conv2d_2/Shape_3:output:01rot_equiv_conv2d_2/strided_slice_3/stack:output:03rot_equiv_conv2d_2/strided_slice_3/stack_1:output:03rot_equiv_conv2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_2/sub_3Sub+rot_equiv_conv2d_2/strided_slice_3:output:0#rot_equiv_conv2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_2/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_2/Const_3:output:0rot_equiv_conv2d_2/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_2/clip_by_value_3Maximum.rot_equiv_conv2d_2/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_2/GatherV2_3GatherV2!rot_equiv_pool2d_1/stack:output:0&rot_equiv_conv2d_2/clip_by_value_3:z:0+rot_equiv_conv2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" «
/rot_equiv_conv2d_2/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0Ь
 rot_equiv_conv2d_2/convolution_3Conv2D&rot_equiv_conv2d_2/GatherV2_3:output:07rot_equiv_conv2d_2/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
д
rot_equiv_conv2d_2/stackPack'rot_equiv_conv2d_2/convolution:output:0)rot_equiv_conv2d_2/convolution_1:output:0)rot_equiv_conv2d_2/convolution_2:output:0)rot_equiv_conv2d_2/convolution_3:output:0*
N*
T0*3
_output_shapes!
:           @*
axis■        ђ
rot_equiv_conv2d_2/ReluRelu!rot_equiv_conv2d_2/stack:output:0*
T0*3
_output_shapes!
:           @ў
)rot_equiv_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0й
rot_equiv_conv2d_2/BiasAddBiasAdd%rot_equiv_conv2d_2/Relu:activations:01rot_equiv_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:           @Z
rot_equiv_pool2d_2/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_2/ShapeShape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d_2/strided_sliceStridedSlice!rot_equiv_pool2d_2/Shape:output:0/rot_equiv_pool2d_2/strided_slice/stack:output:01rot_equiv_pool2d_2/strided_slice/stack_1:output:01rot_equiv_pool2d_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_2/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d_2/subSub)rot_equiv_pool2d_2/strided_slice:output:0!rot_equiv_pool2d_2/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d_2/clip_by_value/MinimumMinimum!rot_equiv_pool2d_2/Const:output:0rot_equiv_pool2d_2/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_2/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d_2/clip_by_valueMaximum,rot_equiv_pool2d_2/clip_by_value/Minimum:z:0+rot_equiv_pool2d_2/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d_2/GatherV2GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0$rot_equiv_pool2d_2/clip_by_value:z:0)rot_equiv_pool2d_2/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @╚
*rot_equiv_pool2d_2/max_pooling2d_2/MaxPoolMaxPool$rot_equiv_pool2d_2/GatherV2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_1Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_2/strided_slice_1StridedSlice#rot_equiv_pool2d_2/Shape_1:output:01rot_equiv_pool2d_2/strided_slice_1/stack:output:03rot_equiv_pool2d_2/strided_slice_1/stack_1:output:03rot_equiv_pool2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_2/sub_1Sub+rot_equiv_pool2d_2/strided_slice_1:output:0#rot_equiv_pool2d_2/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_2/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_2/Const_1:output:0rot_equiv_pool2d_2/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_2/clip_by_value_1Maximum.rot_equiv_pool2d_2/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_2/GatherV2_1GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_1:z:0+rot_equiv_pool2d_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @╠
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1MaxPool&rot_equiv_pool2d_2/GatherV2_1:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_2Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_2/strided_slice_2StridedSlice#rot_equiv_pool2d_2/Shape_2:output:01rot_equiv_pool2d_2/strided_slice_2/stack:output:03rot_equiv_pool2d_2/strided_slice_2/stack_1:output:03rot_equiv_pool2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_2/sub_2Sub+rot_equiv_pool2d_2/strided_slice_2:output:0#rot_equiv_pool2d_2/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_2/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_2/Const_2:output:0rot_equiv_pool2d_2/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_2/clip_by_value_2Maximum.rot_equiv_pool2d_2/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_2/GatherV2_2GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_2:z:0+rot_equiv_pool2d_2/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @╠
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2MaxPool&rot_equiv_pool2d_2/GatherV2_2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_2/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_2/Shape_3Shape#rot_equiv_conv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_2/strided_slice_3StridedSlice#rot_equiv_pool2d_2/Shape_3:output:01rot_equiv_pool2d_2/strided_slice_3/stack:output:03rot_equiv_pool2d_2/strided_slice_3/stack_1:output:03rot_equiv_pool2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_2/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_2/sub_3Sub+rot_equiv_pool2d_2/strided_slice_3:output:0#rot_equiv_pool2d_2/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_2/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_2/Const_3:output:0rot_equiv_pool2d_2/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_2/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_2/clip_by_value_3Maximum.rot_equiv_pool2d_2/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_2/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_2/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_2/GatherV2_3GatherV2#rot_equiv_conv2d_2/BiasAdd:output:0&rot_equiv_pool2d_2/clip_by_value_3:z:0+rot_equiv_pool2d_2/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @╠
,rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3MaxPool&rot_equiv_pool2d_2/GatherV2_3:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
о
rot_equiv_pool2d_2/stackPack3rot_equiv_pool2d_2/max_pooling2d_2/MaxPool:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_1:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_2:output:05rot_equiv_pool2d_2/max_pooling2d_2/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        Z
rot_equiv_conv2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_3/ShapeShape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_conv2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_conv2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_conv2d_3/strided_sliceStridedSlice!rot_equiv_conv2d_3/Shape:output:0/rot_equiv_conv2d_3/strided_slice/stack:output:01rot_equiv_conv2d_3/strided_slice/stack_1:output:01rot_equiv_conv2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_conv2d_3/subSub)rot_equiv_conv2d_3/strided_slice:output:0!rot_equiv_conv2d_3/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_conv2d_3/clip_by_value/MinimumMinimum!rot_equiv_conv2d_3/Const:output:0rot_equiv_conv2d_3/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_conv2d_3/clip_by_valueMaximum,rot_equiv_conv2d_3/clip_by_value/Minimum:z:0+rot_equiv_conv2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Э
rot_equiv_conv2d_3/GatherV2GatherV2!rot_equiv_pool2d_2/stack:output:0$rot_equiv_conv2d_3/clip_by_value:z:0)rot_equiv_conv2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @г
-rot_equiv_conv2d_3/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0У
rot_equiv_conv2d_3/convolutionConv2D$rot_equiv_conv2d_3/GatherV2:output:05rot_equiv_conv2d_3/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_1Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_3/strided_slice_1StridedSlice#rot_equiv_conv2d_3/Shape_1:output:01rot_equiv_conv2d_3/strided_slice_1/stack:output:03rot_equiv_conv2d_3/strided_slice_1/stack_1:output:03rot_equiv_conv2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_3/sub_1Sub+rot_equiv_conv2d_3/strided_slice_1:output:0#rot_equiv_conv2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_3/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_3/Const_1:output:0rot_equiv_conv2d_3/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_3/clip_by_value_1Maximum.rot_equiv_conv2d_3/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_3/GatherV2_1GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_1:z:0+rot_equiv_conv2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @«
/rot_equiv_conv2d_3/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ь
 rot_equiv_conv2d_3/convolution_1Conv2D&rot_equiv_conv2d_3/GatherV2_1:output:07rot_equiv_conv2d_3/convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_2Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_3/strided_slice_2StridedSlice#rot_equiv_conv2d_3/Shape_2:output:01rot_equiv_conv2d_3/strided_slice_2/stack:output:03rot_equiv_conv2d_3/strided_slice_2/stack_1:output:03rot_equiv_conv2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_3/sub_2Sub+rot_equiv_conv2d_3/strided_slice_2:output:0#rot_equiv_conv2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_3/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_3/Const_2:output:0rot_equiv_conv2d_3/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_3/clip_by_value_2Maximum.rot_equiv_conv2d_3/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_3/GatherV2_2GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_2:z:0+rot_equiv_conv2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @«
/rot_equiv_conv2d_3/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ь
 rot_equiv_conv2d_3/convolution_2Conv2D&rot_equiv_conv2d_3/GatherV2_2:output:07rot_equiv_conv2d_3/convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
\
rot_equiv_conv2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_3/Shape_3Shape!rot_equiv_pool2d_2/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_3/strided_slice_3StridedSlice#rot_equiv_conv2d_3/Shape_3:output:01rot_equiv_conv2d_3/strided_slice_3/stack:output:03rot_equiv_conv2d_3/strided_slice_3/stack_1:output:03rot_equiv_conv2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_3/sub_3Sub+rot_equiv_conv2d_3/strided_slice_3:output:0#rot_equiv_conv2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_3/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_3/Const_3:output:0rot_equiv_conv2d_3/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_3/clip_by_value_3Maximum.rot_equiv_conv2d_3/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_3/GatherV2_3GatherV2!rot_equiv_pool2d_2/stack:output:0&rot_equiv_conv2d_3/clip_by_value_3:z:0+rot_equiv_conv2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @«
/rot_equiv_conv2d_3/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_3_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ь
 rot_equiv_conv2d_3/convolution_3Conv2D&rot_equiv_conv2d_3/GatherV2_3:output:07rot_equiv_conv2d_3/convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
д
rot_equiv_conv2d_3/stackPack'rot_equiv_conv2d_3/convolution:output:0)rot_equiv_conv2d_3/convolution_1:output:0)rot_equiv_conv2d_3/convolution_2:output:0)rot_equiv_conv2d_3/convolution_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        ђ
rot_equiv_conv2d_3/ReluRelu!rot_equiv_conv2d_3/stack:output:0*
T0*3
_output_shapes!
:         @ў
)rot_equiv_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0й
rot_equiv_conv2d_3/BiasAddBiasAdd%rot_equiv_conv2d_3/Relu:activations:01rot_equiv_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         @Z
rot_equiv_pool2d_3/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
rot_equiv_pool2d_3/ShapeShape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_pool2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_pool2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_pool2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_pool2d_3/strided_sliceStridedSlice!rot_equiv_pool2d_3/Shape:output:0/rot_equiv_pool2d_3/strided_slice/stack:output:01rot_equiv_pool2d_3/strided_slice/stack_1:output:01rot_equiv_pool2d_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_pool2d_3/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_pool2d_3/subSub)rot_equiv_pool2d_3/strided_slice:output:0!rot_equiv_pool2d_3/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_pool2d_3/clip_by_value/MinimumMinimum!rot_equiv_pool2d_3/Const:output:0rot_equiv_pool2d_3/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_pool2d_3/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_pool2d_3/clip_by_valueMaximum,rot_equiv_pool2d_3/clip_by_value/Minimum:z:0+rot_equiv_pool2d_3/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_pool2d_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Щ
rot_equiv_pool2d_3/GatherV2GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0$rot_equiv_pool2d_3/clip_by_value:z:0)rot_equiv_pool2d_3/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @╚
*rot_equiv_pool2d_3/max_pooling2d_3/MaxPoolMaxPool$rot_equiv_pool2d_3/GatherV2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_1Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_3/strided_slice_1StridedSlice#rot_equiv_pool2d_3/Shape_1:output:01rot_equiv_pool2d_3/strided_slice_1/stack:output:03rot_equiv_pool2d_3/strided_slice_1/stack_1:output:03rot_equiv_pool2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_3/sub_1Sub+rot_equiv_pool2d_3/strided_slice_1:output:0#rot_equiv_pool2d_3/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_3/clip_by_value_1/MinimumMinimum#rot_equiv_pool2d_3/Const_1:output:0rot_equiv_pool2d_3/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_3/clip_by_value_1Maximum.rot_equiv_pool2d_3/clip_by_value_1/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_3/GatherV2_1GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_1:z:0+rot_equiv_pool2d_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @╠
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1MaxPool&rot_equiv_pool2d_3/GatherV2_1:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_2Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_3/strided_slice_2StridedSlice#rot_equiv_pool2d_3/Shape_2:output:01rot_equiv_pool2d_3/strided_slice_2/stack:output:03rot_equiv_pool2d_3/strided_slice_2/stack_1:output:03rot_equiv_pool2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_3/sub_2Sub+rot_equiv_pool2d_3/strided_slice_2:output:0#rot_equiv_pool2d_3/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_3/clip_by_value_2/MinimumMinimum#rot_equiv_pool2d_3/Const_2:output:0rot_equiv_pool2d_3/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_3/clip_by_value_2Maximum.rot_equiv_pool2d_3/clip_by_value_2/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_3/GatherV2_2GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_2:z:0+rot_equiv_pool2d_3/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @╠
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2MaxPool&rot_equiv_pool2d_3/GatherV2_2:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
\
rot_equiv_pool2d_3/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R}
rot_equiv_pool2d_3/Shape_3Shape#rot_equiv_conv2d_3/BiasAdd:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_pool2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_pool2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_pool2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_pool2d_3/strided_slice_3StridedSlice#rot_equiv_pool2d_3/Shape_3:output:01rot_equiv_pool2d_3/strided_slice_3/stack:output:03rot_equiv_pool2d_3/strided_slice_3/stack_1:output:03rot_equiv_pool2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_pool2d_3/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_pool2d_3/sub_3Sub+rot_equiv_pool2d_3/strided_slice_3:output:0#rot_equiv_pool2d_3/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_pool2d_3/clip_by_value_3/MinimumMinimum#rot_equiv_pool2d_3/Const_3:output:0rot_equiv_pool2d_3/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_pool2d_3/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_pool2d_3/clip_by_value_3Maximum.rot_equiv_pool2d_3/clip_by_value_3/Minimum:z:0-rot_equiv_pool2d_3/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_pool2d_3/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ђ
rot_equiv_pool2d_3/GatherV2_3GatherV2#rot_equiv_conv2d_3/BiasAdd:output:0&rot_equiv_pool2d_3/clip_by_value_3:z:0+rot_equiv_pool2d_3/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @╠
,rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3MaxPool&rot_equiv_pool2d_3/GatherV2_3:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
о
rot_equiv_pool2d_3/stackPack3rot_equiv_pool2d_3/max_pooling2d_3/MaxPool:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_1:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_2:output:05rot_equiv_pool2d_3/max_pooling2d_3/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        Z
rot_equiv_conv2d_4/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R y
rot_equiv_conv2d_4/ShapeShape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	y
&rot_equiv_conv2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        {
(rot_equiv_conv2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         r
(rot_equiv_conv2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 rot_equiv_conv2d_4/strided_sliceStridedSlice!rot_equiv_conv2d_4/Shape:output:0/rot_equiv_conv2d_4/strided_slice/stack:output:01rot_equiv_conv2d_4/strided_slice/stack_1:output:01rot_equiv_conv2d_4/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskZ
rot_equiv_conv2d_4/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rї
rot_equiv_conv2d_4/subSub)rot_equiv_conv2d_4/strided_slice:output:0!rot_equiv_conv2d_4/sub/y:output:0*
T0	*
_output_shapes
: Њ
(rot_equiv_conv2d_4/clip_by_value/MinimumMinimum!rot_equiv_conv2d_4/Const:output:0rot_equiv_conv2d_4/sub:z:0*
T0	*
_output_shapes
: d
"rot_equiv_conv2d_4/clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Д
 rot_equiv_conv2d_4/clip_by_valueMaximum,rot_equiv_conv2d_4/clip_by_value/Minimum:z:0+rot_equiv_conv2d_4/clip_by_value/y:output:0*
T0	*
_output_shapes
: k
 rot_equiv_conv2d_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        Э
rot_equiv_conv2d_4/GatherV2GatherV2!rot_equiv_pool2d_3/stack:output:0$rot_equiv_conv2d_4/clip_by_value:z:0)rot_equiv_conv2d_4/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Г
-rot_equiv_conv2d_4/convolution/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0ж
rot_equiv_conv2d_4/convolutionConv2D$rot_equiv_conv2d_4/GatherV2:output:05rot_equiv_conv2d_4/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_1Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_4/strided_slice_1StridedSlice#rot_equiv_conv2d_4/Shape_1:output:01rot_equiv_conv2d_4/strided_slice_1/stack:output:03rot_equiv_conv2d_4/strided_slice_1/stack_1:output:03rot_equiv_conv2d_4/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_4/sub_1Sub+rot_equiv_conv2d_4/strided_slice_1:output:0#rot_equiv_conv2d_4/sub_1/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_4/clip_by_value_1/MinimumMinimum#rot_equiv_conv2d_4/Const_1:output:0rot_equiv_conv2d_4/sub_1:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_4/clip_by_value_1Maximum.rot_equiv_conv2d_4/clip_by_value_1/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_1/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_4/GatherV2_1GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_1:z:0+rot_equiv_conv2d_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @»
/rot_equiv_conv2d_4/convolution_1/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0№
 rot_equiv_conv2d_4/convolution_1Conv2D&rot_equiv_conv2d_4/GatherV2_1:output:07rot_equiv_conv2d_4/convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_2Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_4/strided_slice_2StridedSlice#rot_equiv_conv2d_4/Shape_2:output:01rot_equiv_conv2d_4/strided_slice_2/stack:output:03rot_equiv_conv2d_4/strided_slice_2/stack_1:output:03rot_equiv_conv2d_4/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_4/sub_2Sub+rot_equiv_conv2d_4/strided_slice_2:output:0#rot_equiv_conv2d_4/sub_2/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_4/clip_by_value_2/MinimumMinimum#rot_equiv_conv2d_4/Const_2:output:0rot_equiv_conv2d_4/sub_2:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_4/clip_by_value_2Maximum.rot_equiv_conv2d_4/clip_by_value_2/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_2/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_4/GatherV2_2GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_2:z:0+rot_equiv_conv2d_4/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @»
/rot_equiv_conv2d_4/convolution_2/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0№
 rot_equiv_conv2d_4/convolution_2Conv2D&rot_equiv_conv2d_4/GatherV2_2:output:07rot_equiv_conv2d_4/convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
\
rot_equiv_conv2d_4/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R{
rot_equiv_conv2d_4/Shape_3Shape!rot_equiv_pool2d_3/stack:output:0*
T0*
_output_shapes
:*
out_type0	{
(rot_equiv_conv2d_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        }
*rot_equiv_conv2d_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         t
*rot_equiv_conv2d_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"rot_equiv_conv2d_4/strided_slice_3StridedSlice#rot_equiv_conv2d_4/Shape_3:output:01rot_equiv_conv2d_4/strided_slice_3/stack:output:03rot_equiv_conv2d_4/strided_slice_3/stack_1:output:03rot_equiv_conv2d_4/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask\
rot_equiv_conv2d_4/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rњ
rot_equiv_conv2d_4/sub_3Sub+rot_equiv_conv2d_4/strided_slice_3:output:0#rot_equiv_conv2d_4/sub_3/y:output:0*
T0	*
_output_shapes
: Ў
*rot_equiv_conv2d_4/clip_by_value_3/MinimumMinimum#rot_equiv_conv2d_4/Const_3:output:0rot_equiv_conv2d_4/sub_3:z:0*
T0	*
_output_shapes
: f
$rot_equiv_conv2d_4/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Г
"rot_equiv_conv2d_4/clip_by_value_3Maximum.rot_equiv_conv2d_4/clip_by_value_3/Minimum:z:0-rot_equiv_conv2d_4/clip_by_value_3/y:output:0*
T0	*
_output_shapes
: m
"rot_equiv_conv2d_4/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
■        ■
rot_equiv_conv2d_4/GatherV2_3GatherV2!rot_equiv_pool2d_3/stack:output:0&rot_equiv_conv2d_4/clip_by_value_3:z:0+rot_equiv_conv2d_4/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @»
/rot_equiv_conv2d_4/convolution_3/ReadVariableOpReadVariableOp6rot_equiv_conv2d_4_convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0№
 rot_equiv_conv2d_4/convolution_3Conv2D&rot_equiv_conv2d_4/GatherV2_3:output:07rot_equiv_conv2d_4/convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
Д
rot_equiv_conv2d_4/stackPack'rot_equiv_conv2d_4/convolution:output:0)rot_equiv_conv2d_4/convolution_1:output:0)rot_equiv_conv2d_4/convolution_2:output:0)rot_equiv_conv2d_4/convolution_3:output:0*
N*
T0*4
_output_shapes"
 :         ђ*
axis■        Ђ
rot_equiv_conv2d_4/ReluRelu!rot_equiv_conv2d_4/stack:output:0*
T0*4
_output_shapes"
 :         ђЎ
)rot_equiv_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2rot_equiv_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Й
rot_equiv_conv2d_4/BiasAddBiasAdd%rot_equiv_conv2d_4/Relu:activations:01rot_equiv_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         ђm
"rot_inv_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        ц
rot_inv_pool/MaxMax#rot_equiv_conv2d_4/BiasAdd:output:0+rot_inv_pool/Max/reduction_indices:output:0*
T0*0
_output_shapes
:         ђ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  ђ
flatten/ReshapeReshaperot_inv_pool/Max:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         ђЂ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype0Є
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:          ё
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0І
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ѓ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ╬

NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp(^rot_equiv_conv2d/BiasAdd/ReadVariableOp ^rot_equiv_conv2d/ReadVariableOp"^rot_equiv_conv2d/ReadVariableOp_1"^rot_equiv_conv2d/ReadVariableOp_2,^rot_equiv_conv2d/convolution/ReadVariableOp*^rot_equiv_conv2d_1/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_1/convolution/ReadVariableOp0^rot_equiv_conv2d_1/convolution_1/ReadVariableOp0^rot_equiv_conv2d_1/convolution_2/ReadVariableOp0^rot_equiv_conv2d_1/convolution_3/ReadVariableOp*^rot_equiv_conv2d_2/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_2/convolution/ReadVariableOp0^rot_equiv_conv2d_2/convolution_1/ReadVariableOp0^rot_equiv_conv2d_2/convolution_2/ReadVariableOp0^rot_equiv_conv2d_2/convolution_3/ReadVariableOp*^rot_equiv_conv2d_3/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_3/convolution/ReadVariableOp0^rot_equiv_conv2d_3/convolution_1/ReadVariableOp0^rot_equiv_conv2d_3/convolution_2/ReadVariableOp0^rot_equiv_conv2d_3/convolution_3/ReadVariableOp*^rot_equiv_conv2d_4/BiasAdd/ReadVariableOp.^rot_equiv_conv2d_4/convolution/ReadVariableOp0^rot_equiv_conv2d_4/convolution_1/ReadVariableOp0^rot_equiv_conv2d_4/convolution_2/ReadVariableOp0^rot_equiv_conv2d_4/convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2R
'rot_equiv_conv2d/BiasAdd/ReadVariableOp'rot_equiv_conv2d/BiasAdd/ReadVariableOp2B
rot_equiv_conv2d/ReadVariableOprot_equiv_conv2d/ReadVariableOp2F
!rot_equiv_conv2d/ReadVariableOp_1!rot_equiv_conv2d/ReadVariableOp_12F
!rot_equiv_conv2d/ReadVariableOp_2!rot_equiv_conv2d/ReadVariableOp_22Z
+rot_equiv_conv2d/convolution/ReadVariableOp+rot_equiv_conv2d/convolution/ReadVariableOp2V
)rot_equiv_conv2d_1/BiasAdd/ReadVariableOp)rot_equiv_conv2d_1/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_1/convolution/ReadVariableOp-rot_equiv_conv2d_1/convolution/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_1/ReadVariableOp/rot_equiv_conv2d_1/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_2/ReadVariableOp/rot_equiv_conv2d_1/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_1/convolution_3/ReadVariableOp/rot_equiv_conv2d_1/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_2/BiasAdd/ReadVariableOp)rot_equiv_conv2d_2/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_2/convolution/ReadVariableOp-rot_equiv_conv2d_2/convolution/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_1/ReadVariableOp/rot_equiv_conv2d_2/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_2/ReadVariableOp/rot_equiv_conv2d_2/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_2/convolution_3/ReadVariableOp/rot_equiv_conv2d_2/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_3/BiasAdd/ReadVariableOp)rot_equiv_conv2d_3/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_3/convolution/ReadVariableOp-rot_equiv_conv2d_3/convolution/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_1/ReadVariableOp/rot_equiv_conv2d_3/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_2/ReadVariableOp/rot_equiv_conv2d_3/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_3/convolution_3/ReadVariableOp/rot_equiv_conv2d_3/convolution_3/ReadVariableOp2V
)rot_equiv_conv2d_4/BiasAdd/ReadVariableOp)rot_equiv_conv2d_4/BiasAdd/ReadVariableOp2^
-rot_equiv_conv2d_4/convolution/ReadVariableOp-rot_equiv_conv2d_4/convolution/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_1/ReadVariableOp/rot_equiv_conv2d_4/convolution_1/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_2/ReadVariableOp/rot_equiv_conv2d_4/convolution_2/ReadVariableOp2b
/rot_equiv_conv2d_4/convolution_3/ReadVariableOp/rot_equiv_conv2d_4/convolution_3/ReadVariableOp:Y U
1
_output_shapes
:         љљ
 
_user_specified_nameinputs
йH
к
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_372346

inputs=
#convolution_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбReadVariableOpбReadVariableOp_1бReadVariableOp_2бconvolution/ReadVariableOpє
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0д
convolutionConv2Dinputs"convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:         јј *
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
valueB"       ╗
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
: Њ
convolution_1Conv2Dinputstranspose:y:0*
T0*1
_output_shapes
:         јј *
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
valueB"      ├
TensorScatterUpdate_1TensorScatterUpdaterange_1:output:0&TensorScatterUpdate_1/indices:output:0&TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:ї
transpose_1	Transposeconvolution_1:output:0TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:         јј H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:ђ
ReverseV2_1	ReverseV2transpose_1:y:0ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:         јј Ђ
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
: џ
convolution_2Conv2DinputsReverseV2_3:output:0*
T0*1
_output_shapes
:         јј *
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
valueB:Є
ReverseV2_4	ReverseV2convolution_2:output:0ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:         јј H
Rank_9Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:Ё
ReverseV2_5	ReverseV2ReverseV2_4:output:0ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:         јј ѓ
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
valueB"       ├
TensorScatterUpdate_2TensorScatterUpdaterange_2:output:0&TensorScatterUpdate_2/indices:output:0&TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:|
ReadVariableOp_2ReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Ѓ
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
: џ
convolution_3Conv2DinputsReverseV2_6:output:0*
T0*1
_output_shapes
:         јј *
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
valueB"      ├
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
valueB:Є
ReverseV2_7	ReverseV2convolution_3:output:0ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:         јј і
transpose_3	TransposeReverseV2_7:output:0TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:         јј Й
stackPackconvolution:output:0ReverseV2_1:output:0ReverseV2_5:output:0transpose_3:y:0*
N*
T0*5
_output_shapes#
!:         јј *
axis■        \
ReluRelustack:output:0*
T0*5
_output_shapes#
!:         јј r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0є
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:         јј m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:         јј │
NoOpNoOp^BiasAdd/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^convolution/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         љљ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_228
convolution/ReadVariableOpconvolution/ReadVariableOp:Y U
1
_output_shapes
:         љљ
 
_user_specified_nameinputs
н
I
-__inference_rot_inv_pool_layer_call_fn_372931

inputs
identity┐
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_370560i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ:\ X
4
_output_shapes"
 :         ђ
 
_user_specified_nameinputs
Љ
е
3__inference_rot_equiv_conv2d_3_layer_call_fn_372711

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_370409{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
ЗC
­
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_372926

inputs>
#convolution_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconvolution/ReadVariableOpбconvolution_1/ReadVariableOpбconvolution_2/ReadVariableOpбconvolution_3/ReadVariableOpG
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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Є
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0░
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Ѕ
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0Х
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Ѕ
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0Х
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Ѕ
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0Х
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
╚
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*4
_output_shapes"
 :         ђ*
axis■        [
ReluRelustack:output:0*
T0*4
_output_shapes"
 :         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ё
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         ђl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :         ђ┘
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
к	
З
C__inference_dense_1_layer_call_and_return_conditional_losses_372987

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_369879

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
└6
h
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_370059

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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        д
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј а
max_pooling2d/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:         GG *
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        г

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ц
max_pooling2d/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:         GG *
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        г

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ц
max_pooling2d/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:         GG *
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        г

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ц
max_pooling2d/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
№
stackPackmax_pooling2d/MaxPool:output:0 max_pooling2d/MaxPool_1:output:0 max_pooling2d/MaxPool_2:output:0 max_pooling2d/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         GG *
axis■        b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:         GG "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         јј :] Y
5
_output_shapes#
!:         јј 
 
_user_specified_nameinputs
Х
џ
+__inference_sequential_layer_call_fn_370885
rot_equiv_conv2d_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@ђ
	unknown_8:	ђ
	unknown_9:	ђ 

unknown_10: 

unknown_11: 

unknown_12:
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_370821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
1
_output_shapes
:         љљ
0
_user_specified_namerot_equiv_conv2d_input
К
_
C__inference_flatten_layer_call_and_return_conditional_losses_370568

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
тC
Ь
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_370270

inputs=
#convolution_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconvolution/ReadVariableOpбconvolution_1/ReadVariableOpбconvolution_2/ReadVariableOpбconvolution_3/ReadVariableOpG
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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" є
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0»
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" ѕ
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0х
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" ѕ
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0х
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" ѕ
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0х
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
К
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:           @*
axis■        Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:           @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ё
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:           @k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:           @┘
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         "" : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:         "" 
 
_user_specified_nameinputs
ћ<
у
F__inference_sequential_layer_call_and_return_conditional_losses_370975
rot_equiv_conv2d_input1
rot_equiv_conv2d_370933: %
rot_equiv_conv2d_370935: 3
rot_equiv_conv2d_1_370939:  '
rot_equiv_conv2d_1_370941: 3
rot_equiv_conv2d_2_370945: @'
rot_equiv_conv2d_2_370947:@3
rot_equiv_conv2d_3_370951:@@'
rot_equiv_conv2d_3_370953:@4
rot_equiv_conv2d_4_370957:@ђ(
rot_equiv_conv2d_4_370959:	ђ
dense_370964:	ђ 
dense_370966:  
dense_1_370969: 
dense_1_370971:
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallб(rot_equiv_conv2d/StatefulPartitionedCallб*rot_equiv_conv2d_1/StatefulPartitionedCallб*rot_equiv_conv2d_2/StatefulPartitionedCallб*rot_equiv_conv2d_3/StatefulPartitionedCallб*rot_equiv_conv2d_4/StatefulPartitionedCall▒
(rot_equiv_conv2d/StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_inputrot_equiv_conv2d_370933rot_equiv_conv2d_370935*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:         јј *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_369992ѓ
 rot_equiv_pool2d/PartitionedCallPartitionedCall1rot_equiv_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_370059╩
*rot_equiv_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)rot_equiv_pool2d/PartitionedCall:output:0rot_equiv_conv2d_1_370939rot_equiv_conv2d_1_370941*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_370131ѕ
"rot_equiv_pool2d_1/PartitionedCallPartitionedCall3rot_equiv_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         "" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_370198╠
*rot_equiv_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_1/PartitionedCall:output:0rot_equiv_conv2d_2_370945rot_equiv_conv2d_2_370947*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_370270ѕ
"rot_equiv_pool2d_2/PartitionedCallPartitionedCall3rot_equiv_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_370337╠
*rot_equiv_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_2/PartitionedCall:output:0rot_equiv_conv2d_3_370951rot_equiv_conv2d_3_370953*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_370409ѕ
"rot_equiv_pool2d_3/PartitionedCallPartitionedCall3rot_equiv_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_370476═
*rot_equiv_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_3/PartitionedCall:output:0rot_equiv_conv2d_4_370957rot_equiv_conv2d_4_370959*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_370548щ
rot_inv_pool/PartitionedCallPartitionedCall3rot_equiv_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_370560┘
flatten/PartitionedCallPartitionedCall%rot_inv_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_370568Ђ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_370964dense_370966*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_370581Ј
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_370969dense_1_370971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_370597w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         у
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall)^rot_equiv_conv2d/StatefulPartitionedCall+^rot_equiv_conv2d_1/StatefulPartitionedCall+^rot_equiv_conv2d_2/StatefulPartitionedCall+^rot_equiv_conv2d_3/StatefulPartitionedCall+^rot_equiv_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2T
(rot_equiv_conv2d/StatefulPartitionedCall(rot_equiv_conv2d/StatefulPartitionedCall2X
*rot_equiv_conv2d_1/StatefulPartitionedCall*rot_equiv_conv2d_1/StatefulPartitionedCall2X
*rot_equiv_conv2d_2/StatefulPartitionedCall*rot_equiv_conv2d_2/StatefulPartitionedCall2X
*rot_equiv_conv2d_3/StatefulPartitionedCall*rot_equiv_conv2d_3/StatefulPartitionedCall2X
*rot_equiv_conv2d_4/StatefulPartitionedCall*rot_equiv_conv2d_4/StatefulPartitionedCall:i e
1
_output_shapes
:         љљ
0
_user_specified_namerot_equiv_conv2d_input
ю

з
A__inference_dense_layer_call_and_return_conditional_losses_372968

inputs1
matmul_readvariableop_resource:	ђ -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Љ
е
3__inference_rot_equiv_conv2d_1_layer_call_fn_372421

inputs!
unknown:  
	unknown_0: 
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_370131{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         EE `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         GG : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         GG 
 
_user_specified_nameinputs
╗
L
0__inference_max_pooling2d_3_layer_call_fn_373022

inputs
identity▄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_369903Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ан
к"
"__inference__traced_restore_373372
file_prefixB
(assignvariableop_rot_equiv_conv2d_kernel: 6
(assignvariableop_1_rot_equiv_conv2d_bias: F
,assignvariableop_2_rot_equiv_conv2d_1_kernel:  8
*assignvariableop_3_rot_equiv_conv2d_1_bias: F
,assignvariableop_4_rot_equiv_conv2d_2_kernel: @8
*assignvariableop_5_rot_equiv_conv2d_2_bias:@F
,assignvariableop_6_rot_equiv_conv2d_3_kernel:@@8
*assignvariableop_7_rot_equiv_conv2d_3_bias:@G
,assignvariableop_8_rot_equiv_conv2d_4_kernel:@ђ9
*assignvariableop_9_rot_equiv_conv2d_4_bias:	ђ3
 assignvariableop_10_dense_kernel:	ђ ,
assignvariableop_11_dense_bias: 4
"assignvariableop_12_dense_1_kernel: .
 assignvariableop_13_dense_1_bias:(
assignvariableop_14_nadam_iter:	 *
 assignvariableop_15_nadam_beta_1: *
 assignvariableop_16_nadam_beta_2: )
assignvariableop_17_nadam_decay: 1
'assignvariableop_18_nadam_learning_rate: 2
(assignvariableop_19_nadam_momentum_cache: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: M
3assignvariableop_24_nadam_rot_equiv_conv2d_kernel_m: ?
1assignvariableop_25_nadam_rot_equiv_conv2d_bias_m: O
5assignvariableop_26_nadam_rot_equiv_conv2d_1_kernel_m:  A
3assignvariableop_27_nadam_rot_equiv_conv2d_1_bias_m: O
5assignvariableop_28_nadam_rot_equiv_conv2d_2_kernel_m: @A
3assignvariableop_29_nadam_rot_equiv_conv2d_2_bias_m:@O
5assignvariableop_30_nadam_rot_equiv_conv2d_3_kernel_m:@@A
3assignvariableop_31_nadam_rot_equiv_conv2d_3_bias_m:@P
5assignvariableop_32_nadam_rot_equiv_conv2d_4_kernel_m:@ђB
3assignvariableop_33_nadam_rot_equiv_conv2d_4_bias_m:	ђ;
(assignvariableop_34_nadam_dense_kernel_m:	ђ 4
&assignvariableop_35_nadam_dense_bias_m: <
*assignvariableop_36_nadam_dense_1_kernel_m: 6
(assignvariableop_37_nadam_dense_1_bias_m:M
3assignvariableop_38_nadam_rot_equiv_conv2d_kernel_v: ?
1assignvariableop_39_nadam_rot_equiv_conv2d_bias_v: O
5assignvariableop_40_nadam_rot_equiv_conv2d_1_kernel_v:  A
3assignvariableop_41_nadam_rot_equiv_conv2d_1_bias_v: O
5assignvariableop_42_nadam_rot_equiv_conv2d_2_kernel_v: @A
3assignvariableop_43_nadam_rot_equiv_conv2d_2_bias_v:@O
5assignvariableop_44_nadam_rot_equiv_conv2d_3_kernel_v:@@A
3assignvariableop_45_nadam_rot_equiv_conv2d_3_bias_v:@P
5assignvariableop_46_nadam_rot_equiv_conv2d_4_kernel_v:@ђB
3assignvariableop_47_nadam_rot_equiv_conv2d_4_bias_v:	ђ;
(assignvariableop_48_nadam_dense_kernel_v:	ђ 4
&assignvariableop_49_nadam_dense_bias_v: <
*assignvariableop_50_nadam_dense_1_kernel_v: 6
(assignvariableop_51_nadam_dense_1_bias_v:
identity_53ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9┬
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*У
valueяB█5B9layer_with_weights-0/filt_base/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/filt_base/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-2/filt_base/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/filt_base/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-4/filt_base/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH┌
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ф
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ж
_output_shapesО
н:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOpAssignVariableOp(assignvariableop_rot_equiv_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_1AssignVariableOp(assignvariableop_1_rot_equiv_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_2AssignVariableOp,assignvariableop_2_rot_equiv_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_3AssignVariableOp*assignvariableop_3_rot_equiv_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_4AssignVariableOp,assignvariableop_4_rot_equiv_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_5AssignVariableOp*assignvariableop_5_rot_equiv_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_6AssignVariableOp,assignvariableop_6_rot_equiv_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_7AssignVariableOp*assignvariableop_7_rot_equiv_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_8AssignVariableOp,assignvariableop_8_rot_equiv_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_9AssignVariableOp*assignvariableop_9_rot_equiv_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:Ј
AssignVariableOp_14AssignVariableOpassignvariableop_14_nadam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_15AssignVariableOp assignvariableop_15_nadam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_16AssignVariableOp assignvariableop_16_nadam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_17AssignVariableOpassignvariableop_17_nadam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_18AssignVariableOp'assignvariableop_18_nadam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_19AssignVariableOp(assignvariableop_19_nadam_momentum_cacheIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_24AssignVariableOp3assignvariableop_24_nadam_rot_equiv_conv2d_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_25AssignVariableOp1assignvariableop_25_nadam_rot_equiv_conv2d_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_26AssignVariableOp5assignvariableop_26_nadam_rot_equiv_conv2d_1_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_27AssignVariableOp3assignvariableop_27_nadam_rot_equiv_conv2d_1_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_28AssignVariableOp5assignvariableop_28_nadam_rot_equiv_conv2d_2_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_29AssignVariableOp3assignvariableop_29_nadam_rot_equiv_conv2d_2_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_30AssignVariableOp5assignvariableop_30_nadam_rot_equiv_conv2d_3_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_31AssignVariableOp3assignvariableop_31_nadam_rot_equiv_conv2d_3_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_32AssignVariableOp5assignvariableop_32_nadam_rot_equiv_conv2d_4_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_33AssignVariableOp3assignvariableop_33_nadam_rot_equiv_conv2d_4_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_34AssignVariableOp(assignvariableop_34_nadam_dense_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_35AssignVariableOp&assignvariableop_35_nadam_dense_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_36AssignVariableOp*assignvariableop_36_nadam_dense_1_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_37AssignVariableOp(assignvariableop_37_nadam_dense_1_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_38AssignVariableOp3assignvariableop_38_nadam_rot_equiv_conv2d_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_39AssignVariableOp1assignvariableop_39_nadam_rot_equiv_conv2d_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_40AssignVariableOp5assignvariableop_40_nadam_rot_equiv_conv2d_1_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_41AssignVariableOp3assignvariableop_41_nadam_rot_equiv_conv2d_1_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_42AssignVariableOp5assignvariableop_42_nadam_rot_equiv_conv2d_2_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_43AssignVariableOp3assignvariableop_43_nadam_rot_equiv_conv2d_2_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_44AssignVariableOp5assignvariableop_44_nadam_rot_equiv_conv2d_3_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_45AssignVariableOp3assignvariableop_45_nadam_rot_equiv_conv2d_3_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_46AssignVariableOp5assignvariableop_46_nadam_rot_equiv_conv2d_4_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_47AssignVariableOp3assignvariableop_47_nadam_rot_equiv_conv2d_4_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_48AssignVariableOp(assignvariableop_48_nadam_dense_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_49AssignVariableOp&assignvariableop_49_nadam_dense_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_50AssignVariableOp*assignvariableop_50_nadam_dense_1_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_51AssignVariableOp(assignvariableop_51_nadam_dense_1_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 К	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: ┤	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_53Identity_53:output:0*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ю

з
A__inference_dense_layer_call_and_return_conditional_losses_370581

inputs1
matmul_readvariableop_resource:	ђ -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ћ
ф
3__inference_rot_equiv_conv2d_4_layer_call_fn_372856

inputs"
unknown:@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_370548|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
и
J
.__inference_max_pooling2d_layer_call_fn_372992

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_369867Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Љ
е
3__inference_rot_equiv_conv2d_2_layer_call_fn_372566

inputs!
unknown: @
	unknown_0:@
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_370270{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         "" : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         "" 
 
_user_specified_nameinputs
С
O
3__inference_rot_equiv_pool2d_3_layer_call_fn_372786

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_370476l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
└6
h
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_372412

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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        д
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј а
max_pooling2d/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:         GG *
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        г

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ц
max_pooling2d/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:         GG *
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        г

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ц
max_pooling2d/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:         GG *
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        г

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_output_shapes
:         јј ц
max_pooling2d/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:         GG *
ksize
*
paddingVALID*
strides
№
stackPackmax_pooling2d/MaxPool:output:0 max_pooling2d/MaxPool_1:output:0 max_pooling2d/MaxPool_2:output:0 max_pooling2d/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         GG *
axis■        b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:         GG "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         јј :] Y
5
_output_shapes#
!:         јј 
 
_user_specified_nameinputs
С;
О
F__inference_sequential_layer_call_and_return_conditional_losses_370821

inputs1
rot_equiv_conv2d_370779: %
rot_equiv_conv2d_370781: 3
rot_equiv_conv2d_1_370785:  '
rot_equiv_conv2d_1_370787: 3
rot_equiv_conv2d_2_370791: @'
rot_equiv_conv2d_2_370793:@3
rot_equiv_conv2d_3_370797:@@'
rot_equiv_conv2d_3_370799:@4
rot_equiv_conv2d_4_370803:@ђ(
rot_equiv_conv2d_4_370805:	ђ
dense_370810:	ђ 
dense_370812:  
dense_1_370815: 
dense_1_370817:
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallб(rot_equiv_conv2d/StatefulPartitionedCallб*rot_equiv_conv2d_1/StatefulPartitionedCallб*rot_equiv_conv2d_2/StatefulPartitionedCallб*rot_equiv_conv2d_3/StatefulPartitionedCallб*rot_equiv_conv2d_4/StatefulPartitionedCallА
(rot_equiv_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsrot_equiv_conv2d_370779rot_equiv_conv2d_370781*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:         јј *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_369992ѓ
 rot_equiv_pool2d/PartitionedCallPartitionedCall1rot_equiv_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_370059╩
*rot_equiv_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall)rot_equiv_pool2d/PartitionedCall:output:0rot_equiv_conv2d_1_370785rot_equiv_conv2d_1_370787*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         EE *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_370131ѕ
"rot_equiv_pool2d_1/PartitionedCallPartitionedCall3rot_equiv_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         "" * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_370198╠
*rot_equiv_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_1/PartitionedCall:output:0rot_equiv_conv2d_2_370791rot_equiv_conv2d_2_370793*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_370270ѕ
"rot_equiv_pool2d_2/PartitionedCallPartitionedCall3rot_equiv_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_370337╠
*rot_equiv_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_2/PartitionedCall:output:0rot_equiv_conv2d_3_370797rot_equiv_conv2d_3_370799*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_370409ѕ
"rot_equiv_pool2d_3/PartitionedCallPartitionedCall3rot_equiv_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_370476═
*rot_equiv_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+rot_equiv_pool2d_3/PartitionedCall:output:0rot_equiv_conv2d_4_370803rot_equiv_conv2d_4_370805*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_370548щ
rot_inv_pool/PartitionedCallPartitionedCall3rot_equiv_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_370560┘
flatten/PartitionedCallPartitionedCall%rot_inv_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_370568Ђ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_370810dense_370812*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_370581Ј
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_370815dense_1_370817*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_370597w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         у
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall)^rot_equiv_conv2d/StatefulPartitionedCall+^rot_equiv_conv2d_1/StatefulPartitionedCall+^rot_equiv_conv2d_2/StatefulPartitionedCall+^rot_equiv_conv2d_3/StatefulPartitionedCall+^rot_equiv_conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2T
(rot_equiv_conv2d/StatefulPartitionedCall(rot_equiv_conv2d/StatefulPartitionedCall2X
*rot_equiv_conv2d_1/StatefulPartitionedCall*rot_equiv_conv2d_1/StatefulPartitionedCall2X
*rot_equiv_conv2d_2/StatefulPartitionedCall*rot_equiv_conv2d_2/StatefulPartitionedCall2X
*rot_equiv_conv2d_3/StatefulPartitionedCall*rot_equiv_conv2d_3/StatefulPartitionedCall2X
*rot_equiv_conv2d_4/StatefulPartitionedCall*rot_equiv_conv2d_4/StatefulPartitionedCall:Y U
1
_output_shapes
:         љљ
 
_user_specified_nameinputs
тC
Ь
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_372781

inputs=
#convolution_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconvolution/ReadVariableOpбconvolution_1/ReadVariableOpбconvolution_2/ReadVariableOpбconvolution_3/ReadVariableOpG
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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @є
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0»
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @ѕ
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0х
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @ѕ
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0х
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @ѕ
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0х
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
К
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ё
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         @k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:         @┘
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
к6
j
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_372557

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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE б
max_pooling2d_1/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:         "" *
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE д
max_pooling2d_1/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:         "" *
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE д
max_pooling2d_1/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:         "" *
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         EE д
max_pooling2d_1/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:         "" *
ksize
*
paddingVALID*
strides
э
stackPack max_pooling2d_1/MaxPool:output:0"max_pooling2d_1/MaxPool_1:output:0"max_pooling2d_1/MaxPool_2:output:0"max_pooling2d_1/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         "" *
axis■        b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:         "" "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         EE :[ W
3
_output_shapes!
:         EE 
 
_user_specified_nameinputs
і
Њ
$__inference_signature_wrapper_371016
rot_equiv_conv2d_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@ђ
	unknown_8:	ђ
	unknown_9:	ђ 

unknown_10: 

unknown_11: 

unknown_12:
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallrot_equiv_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__wrapped_model_369858o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
1
_output_shapes
:         љљ
0
_user_specified_namerot_equiv_conv2d_input
├
Ћ
(__inference_dense_1_layer_call_fn_372977

inputs
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_370597o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╗
L
0__inference_max_pooling2d_2_layer_call_fn_373012

inputs
identity▄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_369891Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
к6
j
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_372702

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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @б
max_pooling2d_2/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @д
max_pooling2d_2/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @д
max_pooling2d_2/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:           @д
max_pooling2d_2/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
э
stackPack max_pooling2d_2/MaxPool:output:0"max_pooling2d_2/MaxPool_1:output:0"max_pooling2d_2/MaxPool_2:output:0"max_pooling2d_2/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:           @:[ W
3
_output_shapes!
:           @
 
_user_specified_nameinputs
ЗC
­
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_370548

inputs>
#convolution_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconvolution/ReadVariableOpбconvolution_1/ReadVariableOpбconvolution_2/ReadVariableOpбconvolution_3/ReadVariableOpG
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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Є
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0░
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Ѕ
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0Х
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Ѕ
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0Х
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @Ѕ
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0Х
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
╚
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*4
_output_shapes"
 :         ђ*
axis■        [
ReluRelustack:output:0*
T0*4
_output_shapes"
 :         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ё
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :         ђl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :         ђ┘
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_373007

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
є
і
+__inference_sequential_layer_call_fn_371049

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@ђ
	unknown_8:	ђ
	unknown_9:	ђ 

unknown_10: 

unknown_11: 

unknown_12:
identityѕбStatefulPartitionedCall§
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
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_370604o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         љљ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         љљ
 
_user_specified_nameinputs
Ы
d
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_370560

inputs
identity`
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:         ђ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ:\ X
4
_output_shapes"
 :         ђ
 
_user_specified_nameinputs
С
M
1__inference_rot_equiv_pool2d_layer_call_fn_372351

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         GG * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_370059l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:         GG "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         јј :] Y
5
_output_shapes#
!:         јј 
 
_user_specified_nameinputs
йH
к
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_369992

inputs=
#convolution_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбReadVariableOpбReadVariableOp_1бReadVariableOp_2бconvolution/ReadVariableOpє
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0д
convolutionConv2Dinputs"convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:         јј *
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
valueB"       ╗
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
: Њ
convolution_1Conv2Dinputstranspose:y:0*
T0*1
_output_shapes
:         јј *
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
valueB"      ├
TensorScatterUpdate_1TensorScatterUpdaterange_1:output:0&TensorScatterUpdate_1/indices:output:0&TensorScatterUpdate_1/updates:output:0*
T0*
Tindices0*
_output_shapes
:ї
transpose_1	Transposeconvolution_1:output:0TensorScatterUpdate_1:output:0*
T0*1
_output_shapes
:         јј H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB:ђ
ReverseV2_1	ReverseV2transpose_1:y:0ReverseV2_1/axis:output:0*
T0*1
_output_shapes
:         јј Ђ
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
: џ
convolution_2Conv2DinputsReverseV2_3:output:0*
T0*1
_output_shapes
:         јј *
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
valueB:Є
ReverseV2_4	ReverseV2convolution_2:output:0ReverseV2_4/axis:output:0*
T0*1
_output_shapes
:         јј H
Rank_9Const*
_output_shapes
: *
dtype0*
value	B :Z
ReverseV2_5/axisConst*
_output_shapes
:*
dtype0*
valueB:Ё
ReverseV2_5	ReverseV2ReverseV2_4:output:0ReverseV2_5/axis:output:0*
T0*1
_output_shapes
:         јј ѓ
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
valueB"       ├
TensorScatterUpdate_2TensorScatterUpdaterange_2:output:0&TensorScatterUpdate_2/indices:output:0&TensorScatterUpdate_2/updates:output:0*
T0*
Tindices0*
_output_shapes
:|
ReadVariableOp_2ReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: *
dtype0Ѓ
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
: џ
convolution_3Conv2DinputsReverseV2_6:output:0*
T0*1
_output_shapes
:         јј *
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
valueB"      ├
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
valueB:Є
ReverseV2_7	ReverseV2convolution_3:output:0ReverseV2_7/axis:output:0*
T0*1
_output_shapes
:         јј і
transpose_3	TransposeReverseV2_7:output:0TensorScatterUpdate_3:output:0*
T0*1
_output_shapes
:         јј Й
stackPackconvolution:output:0ReverseV2_1:output:0ReverseV2_5:output:0transpose_3:y:0*
N*
T0*5
_output_shapes#
!:         јј *
axis■        \
ReluRelustack:output:0*
T0*5
_output_shapes#
!:         јј r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0є
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:         јј m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:         јј │
NoOpNoOp^BiasAdd/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^convolution/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         љљ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_228
convolution/ReadVariableOpconvolution/ReadVariableOp:Y U
1
_output_shapes
:         љљ
 
_user_specified_nameinputs
к6
j
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_372847

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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @б
max_pooling2d_3/MaxPoolMaxPoolGatherV2:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @д
max_pooling2d_3/MaxPool_1MaxPoolGatherV2_1:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @д
max_pooling2d_3/MaxPool_2MaxPoolGatherV2_2:output:0*/
_output_shapes
:         @*
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         @д
max_pooling2d_3/MaxPool_3MaxPoolGatherV2_3:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
э
stackPack max_pooling2d_3/MaxPool:output:0"max_pooling2d_3/MaxPool_1:output:0"max_pooling2d_3/MaxPool_2:output:0"max_pooling2d_3/MaxPool_3:output:0*
N*
T0*3
_output_shapes!
:         @*
axis■        b
IdentityIdentitystack:output:0*
T0*3
_output_shapes!
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
┬
ћ
&__inference_dense_layer_call_fn_372957

inputs
unknown:	ђ 
	unknown_0: 
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_370581o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╗
L
0__inference_max_pooling2d_1_layer_call_fn_373002

inputs
identity▄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_369879Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_373027

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╣m
└
__inference__traced_save_373206
file_prefix6
2savev2_rot_equiv_conv2d_kernel_read_readvariableop4
0savev2_rot_equiv_conv2d_bias_read_readvariableop8
4savev2_rot_equiv_conv2d_1_kernel_read_readvariableop6
2savev2_rot_equiv_conv2d_1_bias_read_readvariableop8
4savev2_rot_equiv_conv2d_2_kernel_read_readvariableop6
2savev2_rot_equiv_conv2d_2_bias_read_readvariableop8
4savev2_rot_equiv_conv2d_3_kernel_read_readvariableop6
2savev2_rot_equiv_conv2d_3_bias_read_readvariableop8
4savev2_rot_equiv_conv2d_4_kernel_read_readvariableop6
2savev2_rot_equiv_conv2d_4_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_kernel_m_read_readvariableop<
8savev2_nadam_rot_equiv_conv2d_bias_m_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_1_kernel_m_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_1_bias_m_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_2_kernel_m_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_2_bias_m_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_3_kernel_m_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_3_bias_m_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_4_kernel_m_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_4_bias_m_read_readvariableop3
/savev2_nadam_dense_kernel_m_read_readvariableop1
-savev2_nadam_dense_bias_m_read_readvariableop5
1savev2_nadam_dense_1_kernel_m_read_readvariableop3
/savev2_nadam_dense_1_bias_m_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_kernel_v_read_readvariableop<
8savev2_nadam_rot_equiv_conv2d_bias_v_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_1_kernel_v_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_1_bias_v_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_2_kernel_v_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_2_bias_v_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_3_kernel_v_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_3_bias_v_read_readvariableop@
<savev2_nadam_rot_equiv_conv2d_4_kernel_v_read_readvariableop>
:savev2_nadam_rot_equiv_conv2d_4_bias_v_read_readvariableop3
/savev2_nadam_dense_kernel_v_read_readvariableop1
-savev2_nadam_dense_bias_v_read_readvariableop5
1savev2_nadam_dense_1_kernel_v_read_readvariableop3
/savev2_nadam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ┐
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*У
valueяB█5B9layer_with_weights-0/filt_base/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/filt_base/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-2/filt_base/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-3/filt_base/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-4/filt_base/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/filt_base/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/filt_base/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHО
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B У
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_rot_equiv_conv2d_kernel_read_readvariableop0savev2_rot_equiv_conv2d_bias_read_readvariableop4savev2_rot_equiv_conv2d_1_kernel_read_readvariableop2savev2_rot_equiv_conv2d_1_bias_read_readvariableop4savev2_rot_equiv_conv2d_2_kernel_read_readvariableop2savev2_rot_equiv_conv2d_2_bias_read_readvariableop4savev2_rot_equiv_conv2d_3_kernel_read_readvariableop2savev2_rot_equiv_conv2d_3_bias_read_readvariableop4savev2_rot_equiv_conv2d_4_kernel_read_readvariableop2savev2_rot_equiv_conv2d_4_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_nadam_rot_equiv_conv2d_kernel_m_read_readvariableop8savev2_nadam_rot_equiv_conv2d_bias_m_read_readvariableop<savev2_nadam_rot_equiv_conv2d_1_kernel_m_read_readvariableop:savev2_nadam_rot_equiv_conv2d_1_bias_m_read_readvariableop<savev2_nadam_rot_equiv_conv2d_2_kernel_m_read_readvariableop:savev2_nadam_rot_equiv_conv2d_2_bias_m_read_readvariableop<savev2_nadam_rot_equiv_conv2d_3_kernel_m_read_readvariableop:savev2_nadam_rot_equiv_conv2d_3_bias_m_read_readvariableop<savev2_nadam_rot_equiv_conv2d_4_kernel_m_read_readvariableop:savev2_nadam_rot_equiv_conv2d_4_bias_m_read_readvariableop/savev2_nadam_dense_kernel_m_read_readvariableop-savev2_nadam_dense_bias_m_read_readvariableop1savev2_nadam_dense_1_kernel_m_read_readvariableop/savev2_nadam_dense_1_bias_m_read_readvariableop:savev2_nadam_rot_equiv_conv2d_kernel_v_read_readvariableop8savev2_nadam_rot_equiv_conv2d_bias_v_read_readvariableop<savev2_nadam_rot_equiv_conv2d_1_kernel_v_read_readvariableop:savev2_nadam_rot_equiv_conv2d_1_bias_v_read_readvariableop<savev2_nadam_rot_equiv_conv2d_2_kernel_v_read_readvariableop:savev2_nadam_rot_equiv_conv2d_2_bias_v_read_readvariableop<savev2_nadam_rot_equiv_conv2d_3_kernel_v_read_readvariableop:savev2_nadam_rot_equiv_conv2d_3_bias_v_read_readvariableop<savev2_nadam_rot_equiv_conv2d_4_kernel_v_read_readvariableop:savev2_nadam_rot_equiv_conv2d_4_bias_v_read_readvariableop/savev2_nadam_dense_kernel_v_read_readvariableop-savev2_nadam_dense_bias_v_read_readvariableop1savev2_nadam_dense_1_kernel_v_read_readvariableop/savev2_nadam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *C
dtypes9
725	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
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

identity_1Identity_1:output:0*■
_input_shapesВ
ж: : : :  : : @:@:@@:@:@ђ:ђ:	ђ : : :: : : : : : : : : : : : :  : : @:@:@@:@:@ђ:ђ:	ђ : : :: : :  : : @:@:@@:@:@ђ:ђ:	ђ : : :: 2(
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
:@ђ:!


_output_shapes	
:ђ:%!

_output_shapes
:	ђ : 
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@:-!)
'
_output_shapes
:@ђ:!"

_output_shapes	
:ђ:%#!

_output_shapes
:	ђ : $

_output_shapes
: :$% 

_output_shapes

: : &

_output_shapes
::,'(
&
_output_shapes
: : (

_output_shapes
: :,)(
&
_output_shapes
:  : *

_output_shapes
: :,+(
&
_output_shapes
: @: ,

_output_shapes
:@:,-(
&
_output_shapes
:@@: .

_output_shapes
:@:-/)
'
_output_shapes
:@ђ:!0

_output_shapes	
:ђ:%1!

_output_shapes
:	ђ : 2

_output_shapes
: :$3 

_output_shapes

: : 4

_output_shapes
::5

_output_shapes
: 
тC
Ь
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_372491

inputs=
#convolution_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбconvolution/ReadVariableOpбconvolution_1/ReadVariableOpбconvolution_2/ReadVariableOpбconvolution_3/ReadVariableOpG
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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG є
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0»
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG ѕ
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0х
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG ѕ
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0х
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         GG ѕ
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0х
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         EE *
paddingVALID*
strides
К
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:         EE *
axis■        Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:         EE r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0ё
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         EE k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:         EE ┘
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         GG : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:         GG 
 
_user_specified_nameinputs
▓
D
(__inference_flatten_layer_call_fn_372942

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_370568a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
тC
Ь
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_372636

inputs=
#convolution_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconvolution/ReadVariableOpбconvolution_1/ReadVariableOpбconvolution_2/ReadVariableOpбconvolution_3/ReadVariableOpG
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
■        h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
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
■        ц
GatherV2GatherV2inputsclip_by_value:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" є
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0»
convolutionConv2DGatherV2:output:0"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
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
■        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_1GatherV2inputsclip_by_value_1:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" ѕ
convolution_1/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0х
convolution_1Conv2DGatherV2_1:output:0$convolution_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
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
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_2GatherV2inputsclip_by_value_2:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" ѕ
convolution_2/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0х
convolution_2Conv2DGatherV2_2:output:0$convolution_2/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
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
■        j
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
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
■        ф

GatherV2_3GatherV2inputsclip_by_value_3:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*/
_output_shapes
:         "" ѕ
convolution_3/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0х
convolution_3Conv2DGatherV2_3:output:0$convolution_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingVALID*
strides
К
stackPackconvolution:output:0convolution_1:output:0convolution_2:output:0convolution_3:output:0*
N*
T0*3
_output_shapes!
:           @*
axis■        Z
ReluRelustack:output:0*
T0*3
_output_shapes!
:           @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ё
BiasAddBiasAddRelu:activations:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:           @k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:           @┘
NoOpNoOp^BiasAdd/ReadVariableOp^convolution/ReadVariableOp^convolution_1/ReadVariableOp^convolution_2/ReadVariableOp^convolution_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         "" : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp2<
convolution_1/ReadVariableOpconvolution_1/ReadVariableOp2<
convolution_2/ReadVariableOpconvolution_2/ReadVariableOp2<
convolution_3/ReadVariableOpconvolution_3/ReadVariableOp:[ W
3
_output_shapes!
:         "" 
 
_user_specified_nameinputs"┐L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*м
serving_defaultЙ
c
rot_equiv_conv2d_inputI
(serving_default_rot_equiv_conv2d_input:0         љљ;
dense_10
StatefulPartitionedCall:0         tensorflow/serving/predict:│ь
╚
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
signatures"
_tf_keras_sequential
Й
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	filt_base
bias"
_tf_keras_layer
»
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%pool"
_tf_keras_layer
Й
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,	filt_base
-bias"
_tf_keras_layer
»
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4pool"
_tf_keras_layer
Й
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;	filt_base
<bias"
_tf_keras_layer
»
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cpool"
_tf_keras_layer
Й
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
J	filt_base
Kbias"
_tf_keras_layer
»
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Rpool"
_tf_keras_layer
Й
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Y	filt_base
Zbias"
_tf_keras_layer
Ц
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

mkernel
nbias"
_tf_keras_layer
╗
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias"
_tf_keras_layer
є
0
1
,2
-3
;4
<5
J6
K7
Y8
Z9
m10
n11
u12
v13"
trackable_list_wrapper
є
0
1
,2
-3
;4
<5
J6
K7
Y8
Z9
m10
n11
u12
v13"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Р
|trace_0
}trace_1
~trace_2
trace_32э
+__inference_sequential_layer_call_fn_370635
+__inference_sequential_layer_call_fn_371049
+__inference_sequential_layer_call_fn_371082
+__inference_sequential_layer_call_fn_370885└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 z|trace_0z}trace_1z~trace_2ztrace_3
о
ђtrace_0
Ђtrace_1
ѓtrace_2
Ѓtrace_32с
F__inference_sequential_layer_call_and_return_conditional_losses_371670
F__inference_sequential_layer_call_and_return_conditional_losses_372258
F__inference_sequential_layer_call_and_return_conditional_losses_370930
F__inference_sequential_layer_call_and_return_conditional_losses_370975└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 zђtrace_0zЂtrace_1zѓtrace_2zЃtrace_3
█Bп
!__inference__wrapped_model_369858rot_equiv_conv2d_input"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ё
	ёiter
Ёbeta_1
єbeta_2

Єdecay
ѕlearning_rate
Ѕmomentum_cachemЦmд,mД-mе;mЕ<mфJmФKmгYmГZm«mm»nm░um▒vm▓v│v┤,vх-vХ;vи<vИJv╣Kv║Yv╗Zv╝mvйnvЙuv┐vv└"
	optimizer
-
іserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Іnon_trainable_variables
їlayers
Їmetrics
 јlayer_regularization_losses
Јlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
э
љtrace_02п
1__inference_rot_equiv_conv2d_layer_call_fn_372267б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zљtrace_0
њ
Љtrace_02з
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_372346б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЉtrace_0
1:/ 2rot_equiv_conv2d/kernel
#:! 2rot_equiv_conv2d/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
њnon_trainable_variables
Њlayers
ћmetrics
 Ћlayer_regularization_losses
ќlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
э
Ќtrace_02п
1__inference_rot_equiv_pool2d_layer_call_fn_372351б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЌtrace_0
њ
ўtrace_02з
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_372412б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zўtrace_0
Ф
Ў	variables
џtrainable_variables
Џregularization_losses
ю	keras_api
Ю__call__
+ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ъnon_trainable_variables
аlayers
Аmetrics
 бlayer_regularization_losses
Бlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
щ
цtrace_02┌
3__inference_rot_equiv_conv2d_1_layer_call_fn_372421б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zцtrace_0
ћ
Цtrace_02ш
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_372491б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЦtrace_0
3:1  2rot_equiv_conv2d_1/kernel
%:# 2rot_equiv_conv2d_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
дnon_trainable_variables
Дlayers
еmetrics
 Еlayer_regularization_losses
фlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
щ
Фtrace_02┌
3__inference_rot_equiv_pool2d_1_layer_call_fn_372496б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zФtrace_0
ћ
гtrace_02ш
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_372557б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zгtrace_0
Ф
Г	variables
«trainable_variables
»regularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"
_tf_keras_layer
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
│non_trainable_variables
┤layers
хmetrics
 Хlayer_regularization_losses
иlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
щ
Иtrace_02┌
3__inference_rot_equiv_conv2d_2_layer_call_fn_372566б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zИtrace_0
ћ
╣trace_02ш
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_372636б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╣trace_0
3:1 @2rot_equiv_conv2d_2/kernel
%:#@2rot_equiv_conv2d_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
║non_trainable_variables
╗layers
╝metrics
 йlayer_regularization_losses
Йlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
щ
┐trace_02┌
3__inference_rot_equiv_pool2d_2_layer_call_fn_372641б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┐trace_0
ћ
└trace_02ш
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_372702б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z└trace_0
Ф
┴	variables
┬trainable_variables
├regularization_losses
─	keras_api
┼__call__
+к&call_and_return_all_conditional_losses"
_tf_keras_layer
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Кnon_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
щ
╠trace_02┌
3__inference_rot_equiv_conv2d_3_layer_call_fn_372711б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╠trace_0
ћ
═trace_02ш
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_372781б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z═trace_0
3:1@@2rot_equiv_conv2d_3/kernel
%:#@2rot_equiv_conv2d_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╬non_trainable_variables
¤layers
лmetrics
 Лlayer_regularization_losses
мlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
щ
Мtrace_02┌
3__inference_rot_equiv_pool2d_3_layer_call_fn_372786б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zМtrace_0
ћ
нtrace_02ш
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_372847б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zнtrace_0
Ф
Н	variables
оtrainable_variables
Оregularization_losses
п	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses"
_tf_keras_layer
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
█non_trainable_variables
▄layers
Пmetrics
 яlayer_regularization_losses
▀layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
щ
Яtrace_02┌
3__inference_rot_equiv_conv2d_4_layer_call_fn_372856б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЯtrace_0
ћ
рtrace_02ш
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_372926б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zрtrace_0
4:2@ђ2rot_equiv_conv2d_4/kernel
&:$ђ2rot_equiv_conv2d_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Рnon_trainable_variables
сlayers
Сmetrics
 тlayer_regularization_losses
Тlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
з
уtrace_02н
-__inference_rot_inv_pool_layer_call_fn_372931б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zуtrace_0
ј
Уtrace_02№
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_372937б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zУtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
жnon_trainable_variables
Жlayers
вmetrics
 Вlayer_regularization_losses
ьlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
Ь
Ьtrace_02¤
(__inference_flatten_layer_call_fn_372942б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЬtrace_0
Ѕ
№trace_02Ж
C__inference_flatten_layer_call_and_return_conditional_losses_372948б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z№trace_0
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
­non_trainable_variables
ыlayers
Ыmetrics
 зlayer_regularization_losses
Зlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
В
шtrace_02═
&__inference_dense_layer_call_fn_372957б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zшtrace_0
Є
Шtrace_02У
A__inference_dense_layer_call_and_return_conditional_losses_372968б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zШtrace_0
:	ђ 2dense/kernel
: 2
dense/bias
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
эnon_trainable_variables
Эlayers
щmetrics
 Щlayer_regularization_losses
чlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
Ь
Чtrace_02¤
(__inference_dense_1_layer_call_fn_372977б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЧtrace_0
Ѕ
§trace_02Ж
C__inference_dense_1_layer_call_and_return_conditional_losses_372987б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z§trace_0
 : 2dense_1/kernel
:2dense_1/bias
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
■0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЇBі
+__inference_sequential_layer_call_fn_370635rot_equiv_conv2d_input"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
§BЩ
+__inference_sequential_layer_call_fn_371049inputs"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
§BЩ
+__inference_sequential_layer_call_fn_371082inputs"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ЇBі
+__inference_sequential_layer_call_fn_370885rot_equiv_conv2d_input"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ўBЋ
F__inference_sequential_layer_call_and_return_conditional_losses_371670inputs"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ўBЋ
F__inference_sequential_layer_call_and_return_conditional_losses_372258inputs"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
еBЦ
F__inference_sequential_layer_call_and_return_conditional_losses_370930rot_equiv_conv2d_input"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
еBЦ
F__inference_sequential_layer_call_and_return_conditional_losses_370975rot_equiv_conv2d_input"└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
┌BО
$__inference_signature_wrapper_371016rot_equiv_conv2d_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
тBР
1__inference_rot_equiv_conv2d_layer_call_fn_372267inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_372346inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBР
1__inference_rot_equiv_pool2d_layer_call_fn_372351inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_372412inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ђnon_trainable_variables
Ђlayers
ѓmetrics
 Ѓlayer_regularization_losses
ёlayer_metrics
Ў	variables
џtrainable_variables
Џregularization_losses
Ю__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
З
Ёtrace_02Н
.__inference_max_pooling2d_layer_call_fn_372992б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЁtrace_0
Ј
єtrace_02­
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_372997б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zєtrace_0
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
уBС
3__inference_rot_equiv_conv2d_1_layer_call_fn_372421inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_372491inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBС
3__inference_rot_equiv_pool2d_1_layer_call_fn_372496inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_372557inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Єnon_trainable_variables
ѕlayers
Ѕmetrics
 іlayer_regularization_losses
Іlayer_metrics
Г	variables
«trainable_variables
»regularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
Ш
їtrace_02О
0__inference_max_pooling2d_1_layer_call_fn_373002б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zїtrace_0
Љ
Їtrace_02Ы
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_373007б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЇtrace_0
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
уBС
3__inference_rot_equiv_conv2d_2_layer_call_fn_372566inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_372636inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBС
3__inference_rot_equiv_pool2d_2_layer_call_fn_372641inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_372702inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
јnon_trainable_variables
Јlayers
љmetrics
 Љlayer_regularization_losses
њlayer_metrics
┴	variables
┬trainable_variables
├regularization_losses
┼__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
Ш
Њtrace_02О
0__inference_max_pooling2d_2_layer_call_fn_373012б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЊtrace_0
Љ
ћtrace_02Ы
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_373017б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zћtrace_0
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
уBС
3__inference_rot_equiv_conv2d_3_layer_call_fn_372711inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_372781inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
'
R0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBС
3__inference_rot_equiv_pool2d_3_layer_call_fn_372786inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_372847inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ћnon_trainable_variables
ќlayers
Ќmetrics
 ўlayer_regularization_losses
Ўlayer_metrics
Н	variables
оtrainable_variables
Оregularization_losses
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
Ш
џtrace_02О
0__inference_max_pooling2d_3_layer_call_fn_373022б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zџtrace_0
Љ
Џtrace_02Ы
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_373027б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЏtrace_0
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
уBС
3__inference_rot_equiv_conv2d_4_layer_call_fn_372856inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_372926inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
рBя
-__inference_rot_inv_pool_layer_call_fn_372931inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_372937inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▄B┘
(__inference_flatten_layer_call_fn_372942inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_flatten_layer_call_and_return_conditional_losses_372948inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
┌BО
&__inference_dense_layer_call_fn_372957inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
шBЫ
A__inference_dense_layer_call_and_return_conditional_losses_372968inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▄B┘
(__inference_dense_1_layer_call_fn_372977inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_dense_1_layer_call_and_return_conditional_losses_372987inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
ю	variables
Ю	keras_api

ъtotal

Ъcount"
_tf_keras_metric
c
а	variables
А	keras_api

бtotal

Бcount
ц
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
РB▀
.__inference_max_pooling2d_layer_call_fn_372992inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
§BЩ
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_372997inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
СBр
0__inference_max_pooling2d_1_layer_call_fn_373002inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_373007inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
СBр
0__inference_max_pooling2d_2_layer_call_fn_373012inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_373017inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
СBр
0__inference_max_pooling2d_3_layer_call_fn_373022inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_373027inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
ъ0
Ъ1"
trackable_list_wrapper
.
ю	variables"
_generic_user_object
:  (2total
:  (2count
0
б0
Б1"
trackable_list_wrapper
.
а	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
7:5 2Nadam/rot_equiv_conv2d/kernel/m
):' 2Nadam/rot_equiv_conv2d/bias/m
9:7  2!Nadam/rot_equiv_conv2d_1/kernel/m
+:) 2Nadam/rot_equiv_conv2d_1/bias/m
9:7 @2!Nadam/rot_equiv_conv2d_2/kernel/m
+:)@2Nadam/rot_equiv_conv2d_2/bias/m
9:7@@2!Nadam/rot_equiv_conv2d_3/kernel/m
+:)@2Nadam/rot_equiv_conv2d_3/bias/m
::8@ђ2!Nadam/rot_equiv_conv2d_4/kernel/m
,:*ђ2Nadam/rot_equiv_conv2d_4/bias/m
%:#	ђ 2Nadam/dense/kernel/m
: 2Nadam/dense/bias/m
&:$ 2Nadam/dense_1/kernel/m
 :2Nadam/dense_1/bias/m
7:5 2Nadam/rot_equiv_conv2d/kernel/v
):' 2Nadam/rot_equiv_conv2d/bias/v
9:7  2!Nadam/rot_equiv_conv2d_1/kernel/v
+:) 2Nadam/rot_equiv_conv2d_1/bias/v
9:7 @2!Nadam/rot_equiv_conv2d_2/kernel/v
+:)@2Nadam/rot_equiv_conv2d_2/bias/v
9:7@@2!Nadam/rot_equiv_conv2d_3/kernel/v
+:)@2Nadam/rot_equiv_conv2d_3/bias/v
::8@ђ2!Nadam/rot_equiv_conv2d_4/kernel/v
,:*ђ2Nadam/rot_equiv_conv2d_4/bias/v
%:#	ђ 2Nadam/dense/kernel/v
: 2Nadam/dense/bias/v
&:$ 2Nadam/dense_1/kernel/v
 :2Nadam/dense_1/bias/v┤
!__inference__wrapped_model_369858ј,-;<JKYZmnuvIбF
?б<
:і7
rot_equiv_conv2d_input         љљ
ф "1ф.
,
dense_1!і
dense_1         Б
C__inference_dense_1_layer_call_and_return_conditional_losses_372987\uv/б,
%б"
 і
inputs          
ф "%б"
і
0         
џ {
(__inference_dense_1_layer_call_fn_372977Ouv/б,
%б"
 і
inputs          
ф "і         б
A__inference_dense_layer_call_and_return_conditional_losses_372968]mn0б-
&б#
!і
inputs         ђ
ф "%б"
і
0          
џ z
&__inference_dense_layer_call_fn_372957Pmn0б-
&б#
!і
inputs         ђ
ф "і          Е
C__inference_flatten_layer_call_and_return_conditional_losses_372948b8б5
.б+
)і&
inputs         ђ
ф "&б#
і
0         ђ
џ Ђ
(__inference_flatten_layer_call_fn_372942U8б5
.б+
)і&
inputs         ђ
ф "і         ђЬ
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_373007ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_1_layer_call_fn_373002ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_373017ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_2_layer_call_fn_373012ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_373027ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_3_layer_call_fn_373022ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    В
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_372997ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ─
.__inference_max_pooling2d_layer_call_fn_372992ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    к
N__inference_rot_equiv_conv2d_1_layer_call_and_return_conditional_losses_372491t,-;б8
1б.
,і)
inputs         GG 
ф "1б.
'і$
0         EE 
џ ъ
3__inference_rot_equiv_conv2d_1_layer_call_fn_372421g,-;б8
1б.
,і)
inputs         GG 
ф "$і!         EE к
N__inference_rot_equiv_conv2d_2_layer_call_and_return_conditional_losses_372636t;<;б8
1б.
,і)
inputs         "" 
ф "1б.
'і$
0           @
џ ъ
3__inference_rot_equiv_conv2d_2_layer_call_fn_372566g;<;б8
1б.
,і)
inputs         "" 
ф "$і!           @к
N__inference_rot_equiv_conv2d_3_layer_call_and_return_conditional_losses_372781tJK;б8
1б.
,і)
inputs         @
ф "1б.
'і$
0         @
џ ъ
3__inference_rot_equiv_conv2d_3_layer_call_fn_372711gJK;б8
1б.
,і)
inputs         @
ф "$і!         @К
N__inference_rot_equiv_conv2d_4_layer_call_and_return_conditional_losses_372926uYZ;б8
1б.
,і)
inputs         @
ф "2б/
(і%
0         ђ
џ Ъ
3__inference_rot_equiv_conv2d_4_layer_call_fn_372856hYZ;б8
1б.
,і)
inputs         @
ф "%і"         ђ─
L__inference_rot_equiv_conv2d_layer_call_and_return_conditional_losses_372346t9б6
/б,
*і'
inputs         љљ
ф "3б0
)і&
0         јј 
џ ю
1__inference_rot_equiv_conv2d_layer_call_fn_372267g9б6
/б,
*і'
inputs         љљ
ф "&і#         јј ┬
N__inference_rot_equiv_pool2d_1_layer_call_and_return_conditional_losses_372557p;б8
1б.
,і)
inputs         EE 
ф "1б.
'і$
0         "" 
џ џ
3__inference_rot_equiv_pool2d_1_layer_call_fn_372496c;б8
1б.
,і)
inputs         EE 
ф "$і!         "" ┬
N__inference_rot_equiv_pool2d_2_layer_call_and_return_conditional_losses_372702p;б8
1б.
,і)
inputs           @
ф "1б.
'і$
0         @
џ џ
3__inference_rot_equiv_pool2d_2_layer_call_fn_372641c;б8
1б.
,і)
inputs           @
ф "$і!         @┬
N__inference_rot_equiv_pool2d_3_layer_call_and_return_conditional_losses_372847p;б8
1б.
,і)
inputs         @
ф "1б.
'і$
0         @
џ џ
3__inference_rot_equiv_pool2d_3_layer_call_fn_372786c;б8
1б.
,і)
inputs         @
ф "$і!         @┬
L__inference_rot_equiv_pool2d_layer_call_and_return_conditional_losses_372412r=б:
3б0
.і+
inputs         јј 
ф "1б.
'і$
0         GG 
џ џ
1__inference_rot_equiv_pool2d_layer_call_fn_372351e=б:
3б0
.і+
inputs         јј 
ф "$і!         GG ║
H__inference_rot_inv_pool_layer_call_and_return_conditional_losses_372937n<б9
2б/
-і*
inputs         ђ
ф ".б+
$і!
0         ђ
џ њ
-__inference_rot_inv_pool_layer_call_fn_372931a<б9
2б/
-і*
inputs         ђ
ф "!і         ђН
F__inference_sequential_layer_call_and_return_conditional_losses_370930і,-;<JKYZmnuvQбN
GбD
:і7
rot_equiv_conv2d_input         љљ
p 

 
ф "%б"
і
0         
џ Н
F__inference_sequential_layer_call_and_return_conditional_losses_370975і,-;<JKYZmnuvQбN
GбD
:і7
rot_equiv_conv2d_input         љљ
p

 
ф "%б"
і
0         
џ ─
F__inference_sequential_layer_call_and_return_conditional_losses_371670z,-;<JKYZmnuvAб>
7б4
*і'
inputs         љљ
p 

 
ф "%б"
і
0         
џ ─
F__inference_sequential_layer_call_and_return_conditional_losses_372258z,-;<JKYZmnuvAб>
7б4
*і'
inputs         љљ
p

 
ф "%б"
і
0         
џ г
+__inference_sequential_layer_call_fn_370635},-;<JKYZmnuvQбN
GбD
:і7
rot_equiv_conv2d_input         љљ
p 

 
ф "і         г
+__inference_sequential_layer_call_fn_370885},-;<JKYZmnuvQбN
GбD
:і7
rot_equiv_conv2d_input         љљ
p

 
ф "і         ю
+__inference_sequential_layer_call_fn_371049m,-;<JKYZmnuvAб>
7б4
*і'
inputs         љљ
p 

 
ф "і         ю
+__inference_sequential_layer_call_fn_371082m,-;<JKYZmnuvAб>
7б4
*і'
inputs         љљ
p

 
ф "і         Л
$__inference_signature_wrapper_371016е,-;<JKYZmnuvcб`
б 
YфV
T
rot_equiv_conv2d_input:і7
rot_equiv_conv2d_input         љљ"1ф.
,
dense_1!і
dense_1         