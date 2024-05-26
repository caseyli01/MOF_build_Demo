# MOF_build_Demo
this is a demo of MOF_build package


# MOF_build

MOF_build package with MOF_build modules, can generate MOF with user-defined linker, suitable to build MOF models quickly for multiple use. 
* UIO66 module: 12-coordinate octahedral node, ditopic linker
*  MIL53 module: M(OH) node, ditopic linker
* MIL53_tetra module: M(OH) node, tetratopic linker

---   
## MD preparation interface
Residues appeared in the final structure will be stored in a subfolder 'Residues' for further forcefield generation.

*integrated with forcefield generator of VeloxChem to generate forcefield for linker \
**run MD with Gromacs
