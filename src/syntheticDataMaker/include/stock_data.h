#ifndef STOCK_DATA_H
#define STOCK_DATA_H

#include <stdlib.h>

// Structure to store sector arguments
typedef struct {
    const char *sector;
    const char *sector_description;
    char **tickers;
    int num_tickers;
    double volatility; // Add volatility field
} SectorArgs;

// Function prototypes
void mkdir_recursive(const char *path);
char **generate_ticker_ids(const char *sector, int num_tickers);
void free_ticker_ids(char **tickers, int num_tickers);
const char *get_sector_description(const char *sector);
int generate_stock_data(const char *ticker, const char *sector, double volatility, int days, int intervals_per_day);

#endif // STOCK_DATA_H

