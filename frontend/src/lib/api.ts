const API_BASE = import.meta.env.VITE_API_URL ?? ''

export const AUTH_TOKEN_KEY = 'mira_auth_token'

export function getAuthToken(): string | null {
  return localStorage.getItem(AUTH_TOKEN_KEY)
}

export function setAuthToken(token: string | null): void {
  if (token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token)
  } else {
    localStorage.removeItem(AUTH_TOKEN_KEY)
  }
}

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const token = getAuthToken()

  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(options?.body instanceof FormData
      ? {}
      : { 'Content-Type': 'application/json' }),
    ...(token
      ? { Authorization: `Bearer ${token}` }
      : {}),
    ...(options?.headers as Record<string, string> | undefined),
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    let detail = response.statusText

    try {
      const body = await response.json()
      detail =
        typeof body.detail === 'string'
          ? body.detail
          : JSON.stringify(body.detail ?? body)
    } catch {
      // The response was not JSON.
    }

    if (
      response.status === 401
      && !path.startsWith('/api/auth/login')
    ) {
      setAuthToken(null)
    }

    throw new Error(
      detail || `Request failed (${response.status})`,
    )
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json() as Promise<T>
}


export type UserRole =
  | 'admin'
  | 'data_steward'
  | 'reviewer'
  | 'auditor'
  | string

export type User = {
  id: number
  email: string
  full_name?: string | null
  role: UserRole
  cpse_id?: number | null
  cpse_short_code?: string | null
  is_active: boolean
  created_at?: string | null
}

export type TokenResponse = {
  access_token: string
  token_type: string
  expires_in: number
  user: User
}

export type CpseOption = {
  id: number
  name: string
  short_code: string
}

export type Material = {
  id: number
  cpse: string
  material_code: string
  description: string
  normalized_description: string
  category: string
  unit?: string | null
  manufacturer?: string | null
  manufacturer_part_number?: string | null
  material_grade?: string | null
  parsed_specifications?: Record<string, unknown>
  other_attributes?: Record<string, unknown>
}

export type MatchScores = {
  text_similarity: number
  semantic_similarity: number
  specification_similarity: number
  material_grade_similarity: number
  other_attributes_similarity: number
  final_score: number
}

export type CriticalCheck = {
  field: string
  status: 'PASS' | 'UNKNOWN' | 'CONFLICT'
  source_value: unknown
  target_value: unknown
  reason: string
}

export type CandidateExplanation = {
  decision?: string
  gate_status?: string
  summary?: string
  best_score?: number | null
  second_best_score?: number | null
  score_margin?: number | null
  evaluated_members_count?: number
  critical_checks?: CriticalCheck[]
  [key: string]: unknown
}

/*
 * This is the shape used internally by the existing React components.
 *
 * The backend now stores flat fields:
 *   ai_decision
 *   final_score
 *   text_similarity
 *   semantic_similarity
 *   specification_similarity
 *   material_grade_similarity
 *   other_attributes_similarity
 *
 * normalizeCandidate() below converts the backend response into this
 * UI shape so existing review components continue to work safely.
 */
export type Candidate = {
  id: number
  source_material_id: number
  target_material_id: number
  source_cpse: string
  target_cpse: string
  source_code: string
  target_code: string
  source_description: string
  target_description: string

  scores: MatchScores
  critical_checks: CriticalCheck[]

  engine_decision:
    | 'HIGH_CONFIDENCE'
    | 'REVIEW'
    | 'DIFFERENT'

  review_status: string
  reviewer_id?: string | null
  reviewer_comments?: string | null
  reviewed_at?: string | null
  created_at?: string

  explanation?: CandidateExplanation
}

type BackendCandidate = {
  id: number
  source_material_id: number
  target_material_id: number
  source_cpse: string
  target_cpse: string
  source_code: string
  target_code: string
  source_description: string
  target_description: string

  text_similarity?: number | null
  semantic_similarity?: number | null
  specification_similarity?: number | null
  material_grade_similarity?: number | null
  other_attributes_similarity?: number | null
  final_score?: number | null

  gate_status?: string | null
  ai_decision?: 'HIGH_CONFIDENCE' | 'REVIEW' | 'DIFFERENT'
  review_status: string
  critical_checks?: CriticalCheck[]
  explanation?: CandidateExplanation

  reviewer_id?: string | null
  reviewer_comments?: string | null
  reviewed_at?: string | null
  created_at?: string
}

function normalizeCandidate(
  candidate: BackendCandidate,
): Candidate {
  const finalScore = Number(
    candidate.final_score ?? 0,
  )

  return {
    id: candidate.id,
    source_material_id: candidate.source_material_id,
    target_material_id: candidate.target_material_id,
    source_cpse: candidate.source_cpse,
    target_cpse: candidate.target_cpse,
    source_code: candidate.source_code,
    target_code: candidate.target_code,
    source_description: candidate.source_description,
    target_description: candidate.target_description,

    scores: {
      text_similarity: Number(
        candidate.text_similarity ?? 0,
      ),
      semantic_similarity: Number(
        candidate.semantic_similarity ?? 0,
      ),
      specification_similarity: Number(
        candidate.specification_similarity ?? 0,
      ),
      material_grade_similarity: Number(
        candidate.material_grade_similarity ?? 0,
      ),
      other_attributes_similarity: Number(
        candidate.other_attributes_similarity ?? 0,
      ),
      final_score: finalScore,
    },

    critical_checks: candidate.critical_checks ?? [],
    engine_decision:
      candidate.ai_decision ?? 'REVIEW',
    review_status: candidate.review_status,
    reviewer_id: candidate.reviewer_id,
    reviewer_comments: candidate.reviewer_comments,
    reviewed_at: candidate.reviewed_at,
    created_at: candidate.created_at,

    explanation: {
      ...(candidate.explanation ?? {}),
      gate_status: candidate.gate_status ?? 'UNKNOWN',
    },
  }
}

export type AnalyticsOverview = {
  total_materials: number
  cpse_count: number
  total_candidate_pairs: number
  high_confidence: number
  review_recommendations?: number
  review_pending: number
  approved: number
  rejected: number
  automation_rate: number | null
  gate_breakdown?: {
    pass: number
    unknown: number
    conflict: number
  }
}

export type CpseBreakdown = {
  cpse: string
  material_count: number
  candidate_pair_involvements: number
}

export type CategoryDistribution = {
  category: string
  count: number
}

export type ScoreBucket = {
  bucket: string
  count: number
}

export type CpseDataQuality = {
  cpse: string
  total_materials: number
  with_parsed_specs: number
  parsed_specs_rate: number
  missing_grade: number
  missing_dimensions: number
  missing_pressure: number
}

export type DataQualityMetrics = {
  total_materials: number
  with_parsed_specs: number
  parsed_specs_rate: number
  missing_description: number
  missing_category: number
  missing_material_grade: number
  missing_dimensions: number
  missing_pressure_rating: number
  parsing_failures: number
  completeness_score: number
  by_cpse_quality: CpseDataQuality[]
}

export type AuditEvent = {
  event_type: string
  candidate_id?: number
  source_code?: string
  target_code?: string
  source_cpse?: string
  target_cpse?: string
  actor?: string
  comments?: string | null
  final_score?: number
  timestamp: string
}

export type MappingEntry = {
  material_id: number
  cpse: string
  material_code: string
  description: string
  category?: string
  material_grade?: string | null
}

export type CommonMaterialRecord = {
  canonical_description: string
  category: string
  canonical_technical_attributes: Record<string, unknown>
  source_materials: Array<{
    material_id: number | null
    cpse: string | null
    material_code: string | null
    description: string | null
    source_file?: string | null
    source_page?: number | null
  }>
  provenance: {
    source_count: number
    material_ids: number[]
  }
  approval_status: string
  critical_unknown_fields: string[]
}

export type Mapping = {
  id: number
  nmc: string
  cnmc?: string
  cpse_mappings: MappingEntry[]
  cluster_size: number
  status: string
  common_material_record: CommonMaterialRecord
  created_at: string
}


export const api = {
  login: (
    credentials: {
      email: string
      password: string
    },
  ) =>
    request<TokenResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    }),

  getMe: () =>
    request<User>('/api/auth/me'),

  refresh: (token: string) =>
    request<TokenResponse>('/api/auth/refresh', {
      method: 'POST',
      body: JSON.stringify({
        access_token: token,
      }),
    }),

  logout: () =>
    request<{
      message: string
      user_id: number
    }>('/api/auth/logout', {
      method: 'POST',
    }),

  listUsers: (
    page = 1,
    pageSize = 50,
    role?: string,
  ) => {
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(pageSize),
    })

    if (role && role !== 'all') {
      params.set('role', role)
    }

    return request<{
      items: User[]
      total: number
      page: number
      page_size: number
    }>(`/api/users?${params.toString()}`)
  },

  createUser: (payload: {
    email: string
    password: string
    full_name?: string
    role: string
    cpse_id?: number | null
  }) =>
    request<User>('/api/users', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getUser: (id: number) =>
    request<User>(`/api/users/${id}`),

  updateUser: (
    id: number,
    payload: {
      full_name?: string
      role?: string
      cpse_id?: number | null
      password?: string
      is_active?: boolean
    },
  ) =>
    request<User>(`/api/users/${id}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  deactivateUser: (id: number) =>
    request<User>(`/api/users/${id}`, {
      method: 'DELETE',
    }),

  listCpses: () =>
    request<CpseOption[]>('/api/users/cpses'),

  health: () =>
    request<{
      status: string
      service: string
    }>('/health'),

  uploadMaterials: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    return request<{
      status: string
      records_ingested: number
      total_materials: number
      milvus?: {
        status: string
        embeddings_requested: number
        error?: string | null
      }
    }>('/api/materials/upload', {
      method: 'POST',
      body: formData,
    })
  },

  listMaterials: (params?: {
    query?: string
    cpse?: string
    category?: string
    skip?: number
    limit?: number
  }) => {
    const search = new URLSearchParams()

    if (params?.query) {
      search.set('query', params.query)
    }

    if (params?.cpse) {
      search.set('cpse', params.cpse)
    }

    if (params?.category) {
      search.set('category', params.category)
    }

    if (params?.skip != null) {
      search.set('skip', String(params.skip))
    }

    if (params?.limit != null) {
      search.set('limit', String(params.limit))
    }

    const query = search.toString()

    return request<{
      total: number
      skip: number
      limit: number
      materials: Material[]
    }>(
      `/api/materials${query ? `?${query}` : ''}`,
    )
  },

  materialsStats: () =>
    request<{
      total_materials: number
      cpse_count: number
      cpse_list: string[]
      category_distribution: Record<
        string,
        number
      >
    }>('/api/materials/stats'),

  runBatchMatching: (overwrite = false) =>
    request<{
      status: string
      materials_processed: number
      candidate_pairs_evaluated: number
      new_candidates_stored: number
      total_candidates: number
      decision_breakdown: Record<
        string,
        number
      >
      gate_breakdown?: Record<string, number>
      vector_search_hits?: number
      vector_search_failures?: number
      elapsed_ms: number
    }>('/api/matching/run-batch', {
      method: 'POST',
      body: JSON.stringify({
        overwrite,
        max_candidates_per_material: 15,
      }),
    }),

  reviewQueue: (skip = 0, limit = 100) =>
    request<{
      total_pending: number
      skip: number
      limit: number
      queue: BackendCandidate[]
    }>(
      `/api/review/queue?skip=${skip}&limit=${limit}`,
    ).then((response) => ({
      ...response,
      queue: response.queue.map(normalizeCandidate),
    })),

  reviewSummary: () =>
    request<{
      total_review_queue: number
      total_reviewable?: number
      pending: number
      approved: number
      rejected: number
      high_confidence: number
      review_recommendations?: number
    }>('/api/review/summary'),

  reviewAction: (
    candidateId: number,
    action: 'APPROVE' | 'REJECT',
    comments?: string,
  ) =>
    request<{
      status: string
      candidate: BackendCandidate
    }>(
      `/api/review/queue/${candidateId}/action`,
      {
        method: 'POST',
        body: JSON.stringify({
          action,
          reviewer_comments: comments ?? null,
        }),
      },
    ).then((response) => ({
      ...response,
      candidate: normalizeCandidate(
        response.candidate,
      ),
    })),

  analyticsOverview: () =>
    request<AnalyticsOverview>(
      '/api/analytics/overview',
    ),

  analyticsByCpse: () =>
    request<{
      cpse_breakdown: CpseBreakdown[]
    }>('/api/analytics/by-cpse'),

  analyticsCategories: () =>
    request<{
      category_distribution: CategoryDistribution[]
    }>('/api/analytics/categories'),

  analyticsScores: () =>
    request<{
      total_candidates: number
      score_histogram: ScoreBucket[]
    }>('/api/analytics/scores'),

  analyticsDataQuality: () =>
    request<DataQualityMetrics>(
      '/api/analytics/data-quality',
    ),

  listMappings: () =>
    request<{
      total: number
      mappings: Mapping[]
    }>('/api/mappings'),

  generateMappings: () =>
    request<{
      status: string
      mappings_created: number
      total_mappings: number
      mappings: Mapping[]
      message?: string
    }>('/api/mappings/generate', {
      method: 'POST',
    }),

  approveMapping: (mappingId: number) =>
    request<{
      status: string
      mapping: Mapping
      message?: string
    }>(
      `/api/mappings/${mappingId}/approve`,
      {
        method: 'POST',
      },
    ),

  exportMappingsFlat: () =>
    request<{
      total_rows: number
      rows: Array<{
        nmc: string
        cnmc?: string
        cpse: string
        cpse_material_code: string
        description: string
        category?: string
        material_grade?: string | null
      }>
    }>('/api/mappings/export/flat'),

  listAudit: (skip = 0, limit = 100) =>
    request<{
      total: number
      events: AuditEvent[]
    }>(
      `/api/audit?skip=${skip}&limit=${limit}`,
    ),

  exportAudit: () =>
    request<{
      total: number
      events: AuditEvent[]
    }>('/api/audit/export'),

  findCnmcCandidates: (material: {
    id?: number
    cpse: string
    material_code: string
    description: string
    category?: string
    unit?: string
    material_grade?: string
  }) =>
    request<{
      material: Material
      total_candidates: number
      best_score?: number | null
      second_best_score?: number | null
      score_margin?: number | null
      candidates: Array<{
        cnmc_code: string
        cnmc_id: number
        mapping_id?: number
        material_type: string
        category: string
        standardized_description: string
        scores: MatchScores
        critical_checks: CriticalCheck[]
        engine_decision:
          | 'HIGH_CONFIDENCE'
          | 'REVIEW'
          | 'DIFFERENT'
        final_score: number
        canonical_score: number
        strongest_member_score?: number | null
        member_count: number
        best_score?: number | null
        second_best_score?: number | null
        score_margin?: number | null
      }>
    }>('/api/matching/cnmc/candidates', {
      method: 'POST',
      body: JSON.stringify({ material }),
    }),

  runCnmcBatchMatching: (
    maxCandidatesPerMaterial = 5,
    minScore = 0.65,
  ) =>
    request<{
      status: string
      materials_evaluated: number
      proposals_generated: number
      proposals: unknown[]
    }>('/api/matching/cnmc/run-batch', {
      method: 'POST',
      body: JSON.stringify({
        max_candidates_per_material:
          maxCandidatesPerMaterial,
        min_score: minScore,
        create_review_candidates: true,
      }),
    }),

  attachMaterialToMapping: (
    mappingId: number,
    materialId: number,
  ) =>
    request<{
      status: string
      mapping: Mapping
    }>(
      `/api/mappings/${mappingId}/attach`,
      {
        method: 'POST',
        body: JSON.stringify({
          material_id: materialId,
        }),
      },
    ),
}


export function formatPercent(
  value: number,
  digits = 0,
): string {
  return `${(value * 100).toFixed(digits)}%`
}

export function formatNumber(
  value: number,
): string {
  return value.toLocaleString()
}

export function formatTimestamp(
  iso: string,
): string {
  const date = new Date(iso)

  if (Number.isNaN(date.getTime())) {
    return iso
  }

  return date.toLocaleString(
    undefined,
    {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    },
  )
}