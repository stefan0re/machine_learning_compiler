	.text
	.file	"hello_assembly.c"
	.globl	hello_assembly                  // -- Begin function hello_assembly
	.p2align	2
	.type	hello_assembly,@function
hello_assembly:                         // @hello_assembly
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	bl	printf
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end0:
	.size	hello_assembly, .Lfunc_end0-hello_assembly
	.cfi_endproc
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Hello Assembly Language!\n"
	.size	.L.str, 26

	.ident	"clang version 19.1.7 (Fedora 19.1.7-3.fc41)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym printf