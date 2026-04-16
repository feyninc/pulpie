Memory Layout in C

The memory layout of a C program refers to how the program's memory is organized during its execution. Understanding the memory layout helps developers manage memory more effectively, debug programs, and avoid common memory-related errors.

The memory is typically divided into the following distinct **memory segments** −

Text segment

Initialized data segment

Uninitialized data segment

Stack

Efficiently managing these memory segments in RAM, which is faster but limited in capacity as compared to the secondary storage, is crucial for preventing segmentation faults and optimizing C program execution.

The following illustration shows how the memory layout is organized, and also depicts how the RAM loads a C program into its different memory segments −

Let us discuss each of these memory segments in detail.

Text Segment

The **textsegment** is also known as the **codesegment**, which generates a **binary file** after compiling the program. This binary file is then used to execute the program by loading it into the RAM. This binary file contains **instructions** that get stored in the text segment of the memory.

The **textsegment** is usually read-only and stored in the lower part of the memory to prevent accidental modification of the code while the program is running.

The size of the **textsegment** determines the number of instructions and the complexity of the program.

Initialized Data Segment

The **initialized data segment** is a type of data segment that stores the **global** and **staticvariables** created by the programmer. This segment is placed just above the text segment of the program.

The **initializeddatasegment** contains global and static variables that have been explicitly initialized by the programmer. For **example**,

// Global variable int a = 10; // Static variable static int b = 20;

This memory segment has read-write permission because the value of a variable can change during program execution.

Example: Initialized Data Segment

The following C program shows how the **initialized data segment** works −

#include<stdio.h> int globalVar1 = 50; char *greet = "Hello World"; const int globalVar2 = 30; int main() { // static variable stored in initialized data segment static int n = 10; // ... printf("Global variables are stored in Initialize Data Segment"); return 0; }

In this code, the variables **globalVar1** and the pointer **greet** are declared outside the scope of the **main**() function, and therefore they are stored in the **read-write** section of the **initializeddatasegment**. However, the global variable **globalVar2** is declared with the keyword **const**, and hence it is stored in the **read**-**only** section of the **initializeddatasegment**. Static variables like **a** are also stored in this part of the memory.

When you run this code, it will produce the following **output** −

Global variables are stored in Initialize Data Segment

Uninitialized Data Segment

The **uninitialized data segment**, also known as the **BSS (Block Started by Symbol)** segment, is a part of a C program's memory layout. When a program is loaded into the memory, space for the BSS segment is allocated by the operating system. Before the execution of the C program begins, the kernel automatically initializes all variables in the BSS segment: arithmetic data types are set to 0, and pointers are set to a null pointer.

The BSS segment contains all the **global** and **static variables** that are not explicitly initialized by the programmer (or initialized with 0). Since the values of these variables can be modified during program execution, the BSS segment has **read-write permission**.

Example: Uninitialized Data Segment

Let's understand the role of uninitialized data segment through the following C program −

#include <stdio.h> // Uninitialized global variable stored in the bss segment int globalVaraible; int main(){ // Uninitialized static variable stored in bss static int staticVariable; printf("Global Variable = %d\n", globalVaraible); printf("Static Variable = %d\n", staticVariable); return 0; }

When you run this code, it will produce the following **output** −

Global Variable = 0 Static Variable = 0

In this C program, both the **static** and **global** variables are **uninitialized**, so they are stored in the BSS segment of the memory layout. Before the program execution begins, the kernel initializes these variables with the value 0.

Heap Segment

The **heap** area begins at the end of the **BSSsegment** and grows upward toward higher memory addresses. It is the memory segment used for **dynamicmemory** allocation during program execution. Whenever additional memory is required, functions like **malloc**() and **calloc**() allocate space from the heap, causing it to grow upward.

The heap is managed by functions such as **malloc**(), **calloc**(), and **free**(), which internally may use system calls like **brk** and **sbrk** to adjust its size.

Since the heap is a shared region, it is also used by all shared libraries and dynamically loaded modules within a process.

Example: Heap Segment

In this C program, we have created a variable of data type **char**, which allocates 1 byte of memory (the size of a **char** in C) at the time of program execution. Since this variable is created dynamically, it is allocated in the heap segment of the memory.

#include <stdio.h> #include <stdlib.h> int main() { // Allocate memory for a single char char *var = (char*)malloc(sizeof(char)); *var = 'A'; // Print the value and the size of the allocated memory printf("Value of dynamically allocated char: %c\n",* var); printf("Size of dynamically allocated char: %zu bytes\n", sizeof(*var)); // Free the dynamically allocated memory free(var); return 0; }

Run the code and check its **output** −

Value of dynamically allocated char: A Size of dynamically allocated char: 1 bytes

Stack Segment

The stack segment follows a **LIFO** (Last In, First Out) structure and usually grows **downward** toward lower memory addresses (though the exact behavior depends on the computer architecture). It grows in the direction **opposite** to the **heap**.

The stack is used to manage **functioncalls** and **localvariables**. Each time a function is called, a stack frame is created, which stores the function’s **localvariables**, parameters, and return address. When the function finishes, its stack frame is removed, following the **LIFO** principle.

Example: Stack Segment

The following example shows how the variables are stored in the stack memory segment −

#include <stdio.h> void display(int x) { int y = 20; printf("Parameter x = %d\n", x); printf("Local variable y = %d\n", y); } int main() { int mainVar = 10; // function call creates new stack frame display(mainVar); return 0; }

Run the code and check its **output** −

Parameter x = 10 Local variable y = 20

When the **main** function starts, its **stack frame** is created storing **mainVar** and the return address.
When the **display** function is called, a **new stack frame** is pushed storing **x** and **y**, and removed once the function ends (LIFO order).

Command-line Arguments

When a C program is executed, any command-line arguments passed to it are also stored in the memory. These arguments are placed in the special memory segment, typically above the stack in the process memory layout.

The command-line arguments are passed to the **main()** function in the form of −

int main(int argc, char *argv[])

Here,

**argc** (argument count): Stores the total number of argument passed, includes the program name.

**argv** (argument vector): It is an array of character pointer (strings), where each element point to a command-line arguments.

Example: argc and argv

Let's understand both arguments (**argc** and **argv**) through a C program:

#include <stdio.h> int main(int argc, char *argv[]) { printf("Total arguments: %d\n", argc); for (int i = 0; i < argc; i++) { printf("Argument %d: %s\n", i, argv[i]); } return 0; }

Following is the **output** of the above code −

Total arguments: 1 Argument 0: /tmp/HqDVg7xJye/main.o

Example: Program to get the Size of Memory Segment

In this example, we create a simple C program layout and use the command below to get the size of each memory segment. To run this, you need a Linux environment. On Windows, you can download and install **MinGW** to use **GCC** and related commands.

#include<stdio.h> int main() { return 0; }

Use the following command to get the size −

gcc file_name.c -o file_name size file_name

~$ gcc program.c -o program ~$ size program text data bss dec hex filename 1418 544 8 1970 7b0 program

Example: Inserting an Uninitialized Global Variable

Inserting an uninitialized global variable increases the size of the Data segment −

#include <stdio.h> int global; int main() { return 0; }

~$ gcc program.c -o program ~$ size program text data bss dec hex filename 1418 548 8 1970 7b0 program

Example: Inserting an Uninitialized Static Variable

If you insert an uninitialized static variable, it increases the occupied space in the **BSS** segment.

#include <stdio.h> int globalVar = 10; int main() { static int staticVar; return 0; }

~$ gcc program.c -o program ~$ size program text data bss dec hex filename 1418 548 12 1970 7b0 program

If you insert a static variable with an initialized value, it will be stored in the data segment.

#include <stdio.h> int globalVar = 10; int main() { static int staticVar; static int a = 5; return 0; }

~$ gcc program.c -o program ~$ size program text data bss dec hex filename 1418 552 8 1970 7b0 program

As we saw in the above programs, if we insert a global variable without initialization, it will be stored in the **BSS** segment.

Conclusion

The memory layout of a C program is divided into distinct segments: text segment, data segment, BSS, heap, and stack. Each segment has a specific role in program execution. The text segment stores code, while the data and BSS segments handle global and static variables. The heap manages dynamic memory, and the stack is used for function calls and local variables.