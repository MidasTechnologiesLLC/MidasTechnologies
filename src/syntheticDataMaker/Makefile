# Makefile for Synthetic Stock Data Generator

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -pthread -Iinclude

# Directories
SRC_DIR = src
INCLUDE_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Files
SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/stock_data.c
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/stock_data.o
TARGET = $(BIN_DIR)/synthetic_data_maker

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@

# Compile object files
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.c $(INCLUDE_DIR)/stock_data.h
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/stock_data.o: $(SRC_DIR)/stock_data.c $(INCLUDE_DIR)/stock_data.h
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean

