#ifndef __ALGO_TYPES_H
#define __ALGO_TYPES_H
// TODO: Add QTYPE definitons in here

// Data populated by flash_dtw()
typedef struct{
  short query_number;   // populated only if flash_dtw() is called with null query and it therefore falls back on query(ies) loaded in an earlier call to segment_and_load_queries()
  float p_value;
  float fdr;            // populated only if threshold specified 
  int match_ordinal;    // populated only if max # ranks and/or FDR threshold specified
  int left_anchor_query;
  int *middle_anchors_query; // populated only if record_anchors == TRUE in call to flash_dtw() 
  int right_anchor_query;
  long left_anchor_subject;
  long *middle_anchors_subject; // populated only if record_anchors == TRUE in call to flash_dtw() 
  long right_anchor_subject;
  int num_anchors;
} match_record;
#endif