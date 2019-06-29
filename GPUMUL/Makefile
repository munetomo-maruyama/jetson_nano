#===============================
# Makefile
#-------------------------------
# Rev.01 2019.05.05 M.Maruyama
#===============================
# Memo
# ${SRCS:.c=.o} means, 
#     take the variable value ${SRCS}, 
#     which is a string composed of words separated by spaces, 
#     and for each word, replace the suffix .c with .o.
# $@ means Target Name
# $< means First File Name of Dependent Files
# $^ means List of Dependent Files
 
PROGNAME := GPUMUL

SRCDIR := src
OBJDIR := obj
SRCS := $(SRCDIR)/main.cu   \
        $(SRCDIR)/do_cpu.cu \
        $(SRCDIR)/do_gpu.cu

NVCC := nvcc
NVCFLAGS := -O3 -gencode arch=compute_53,code=sm_53 --ptxas-options=-v
LDFLAGS :=
LIBS :=

PROG := $(OBJDIR)/$(PROGNAME)
OBJS := $(SRCS:%.cu=$(OBJDIR)/%.o)
DEPS := $(SRCS:%.cu=$(OBJDIR)/%.d)

.PHONY: all clean

all: $(PROG)

$(PROG): $(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR)/%.d:%.cu
	@mkdir -p $(OBJDIR)
	@mkdir -p $(OBJDIR)/$(SRCDIR)
#	@$(NVCC) -M $(NVCFLAGS) $< > $@
	@$(NVCC) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MM" $(NVCFLAGS) -c $< -o $@
	@sed -i '1s/^/$(OBJDIR)\/$(SRCDIR)\//' $@

-include $(DEPS)

$(OBJDIR)/%.o:%.cu
	@mkdir -p $(OBJDIR)
	@mkdir -p $(OBJDIR)/$(SRCDIR)
	$(NVCC) $(NVCFLAGS) -c $< -o $@ 

clean:
	rm -rf $(OBJDIR)/*

#===============================
# End of File
#===============================
