app = Flask(__name__)
register_formula_routes(app)
# Load classifier at startup
classifier = pickle.load(open('odoo_classifier.pkl', 'rb'))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def classify_command(message):
    embedding = embedder.encode([message])
    prediction = classifier.predict(embedding)[0]
    confidence = classifier.predict_proba(embedding)[0].max()
    
    return {
        'is_odoo': prediction == 1,
        'confidence': confidence
    }
# Comprehensive Model Mappings
MODEL_MAPPINGS = {
    'crm.lead': ['lead', 'leads', 'crm', 'opportunity', 'opportunities', 'prospect', 'prospects'],
    'crm.team': ['team', 'teams', 'sales team', 'crm team'],
    'sale.order': ['sale', 'sales', 'order', 'orders', 'quotation', 'quotations', 'quote', 'quotes'],
    'sale.order.line': ['order line', 'order lines', 'sale line', 'sale lines'],
    'purchase.order': ['purchase', 'purchases', 'po', 'purchase order', 'purchase orders'],
    'purchase.order.line': ['purchase line', 'purchase lines', 'po line', 'po lines'],
    'stock.picking': ['picking', 'pickings', 'delivery', 'deliveries', 'transfer', 'transfers', 'shipment', 'shipments'],
    'stock.move': ['stock move', 'stock moves', 'inventory move', 'inventory moves'],
    'stock.warehouse': ['warehouse', 'warehouses', 'stock location', 'storage'],
    'account.move': ['invoice', 'invoices', 'bill', 'bills', 'journal entry', 'journal entries', 'receipt', 'receipts'],
    'account.move.line': ['invoice line', 'invoice lines', 'bill line', 'bill lines'],
    'account.payment': ['payment', 'payments', 'receipt', 'receipts'],
    'hr.employee': ['employee', 'employees', 'staff', 'worker', 'workers', 'team member', 'team members', 'attendance'],
    'hr.attendance': ['attendance', 'attendances', 'check in', 'check out'],
    'hr.leave': ['leave', 'leaves', 'vacation', 'holidays', 'time off'],
    'hr.department': ['department', 'departments', 'division', 'divisions'],
    'project.project': ['project', 'projects', 'job', 'jobs'],
    'project.task': ['task', 'tasks', 'activity', 'activities', 'assignment', 'assignments'],
    'project.milestone': ['milestone', 'milestones', 'goal', 'goals'],
    'res.partner': ['partner', 'partners', 'customer', 'customers', 'client', 'clients', 'contact', 'contacts', 'vendor', 'vendors', 'supplier', 'suppliers'],
    'res.company': ['company', 'companies', 'organization', 'organizations'],
    'product.product': ['product', 'products', 'item', 'items', 'stock', 'inventory'],
    'product.template': ['product template', 'product templates', 'product type', 'product types'],
    'product.category': ['category', 'categories', 'product category', 'product categories'],
    'mrp.production': ['production', 'productions', 'manufacturing', 'manufacturing order', 'manufacturing orders'],
    'mrp.workorder': ['work order', 'work orders', 'workorder', 'workorders'],
    'mrp.bom': ['bom', 'bill of materials', 'recipe', 'recipes'],
    'website.page': ['page', 'pages', 'web page', 'web pages'],
    'website.menu': ['menu', 'menus', 'navigation', 'nav'],
    'helpdesk.ticket': ['ticket', 'tickets', 'support ticket', 'support tickets', 'issue', 'issues'],
    'mail.channel': ['channel', 'channels', 'discussion', 'discussions'],
    'mail.message': ['message', 'messages', 'email', 'emails'],
    'res.users': ['user', 'users', 'account', 'accounts'],
    'res.groups': ['group', 'groups', 'role', 'roles', 'permission', 'permissions'],
    'ir.model': ['model', 'models', 'table', 'tables'],
    'account.analytic.line': ['timesheet', 'timesheets', 'time entry', 'time entries', 'analytic line', 'analytic lines']
    ,'crm.lead': ['lead', 'prospect', 'customer', 'client', 'opportunity', 'quotation', 'quote', 'status'],
    'sale.order': ['order', 'sale', 'quotation', 'quote', 'so', 'sales order'],
    'res.partner': ['partner', 'contact', 'customer', 'vendor', 'supplier'],
    'product.product': ['product', 'item', 'goods', 'service'],
    'account.move': ['invoice', 'bill', 'payment', 'accounting'],
    'stock.picking': ['delivery', 'shipment', 'picking', 'transfer'],
    'hr.employee': ['employee', 'staff', 'worker', 'personnel'],
    'project.project': ['project', 'task', 'milestone'],
    'helpdesk.ticket': ['ticket', 'support', 'issue', 'problem'],
    'purchase.order': ['purchase', 'po', 'buy', 'procurement']

}

# Intent detection patterns
INTENT_PATTERNS = {
    "create": [
        r"create|add|new|make|register|insert|build|generate|establish|form",
        r"i want to (create|add|make)",
        r"can you (create|add|make)",
        r"please (create|add|make)",
        r"need to (create|add|make)",
        r"let's (create|add|make)"
    ],
    "read": [
        r"show|get|fetch|find|search|list|display|view|retrieve|see|check",
        r"what are|what is|who are|who is|where are|where is|when are|when is",
        r"give me|tell me|i want to see|i need to see",
        r"can you show|can you get|can you find",
        r"please show|please get|please find",
        r"list all|show all|get all|fetch all",
        r"total|sum|how many"
    ],
    "update": [
        r"update|edit|modify|change|alter|revise|adjust|fix|correct|amend",
        r"i want to (update|edit|modify|change)",
        r"can you (update|edit|modify|change)",
        r"please (update|edit|modify|change)",
        r"need to (update|edit|modify|change)",
        r"let's (update|edit|modify|change)"
    ],
    "delete": [
        r"delete|remove|drop|eliminate|cancel|destroy|erase|clear",
        r"i want to (delete|remove|drop)",
        r"can you (delete|remove|drop)",
        r"please (delete|remove|drop)",
        r"need to (delete|remove|drop)",
        r"let's (delete|remove|drop)"
    ],
    "count": [
        r"count|how many|number of|total|sum|quantity",
        r"how many are|how many do|how much",
        r"what's the count|what's the total|what's the number",
        r"total.*leads|total.*customers|total.*orders"
    ],
    "report": [
        r"report|analytics|analysis|dashboard|summary|overview",
        r"show me report|generate report|create report",
        r"monthly report|weekly report|daily report|yearly report"
    ]
}

def detect_intent(message):
    """Detect user intent from message using pattern matching"""
    message = message.lower()
    # 👇 Add this right after message_lower = message.lower()
    if "quotation" in message.lower or "quote" in message.lower:
        return "sale.order"

    # Check for count/total keywords first
    if any(keyword in message for keyword in ['total', 'count', 'how many', 'number of']):
        return "count"

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, message):
                return intent

    return "read"  # Default to read

def preprocess_message(message):
    """Clean and normalize user message for better processing"""
    message = message.lower().strip()
    message = message.replace("today's", "todays")
    message = message.replace(" as a ", " as ")
    message = message.replace(" as ", " ")
    message = message.replace(" id ", " ")
    return message



def detect_model(message):
    """Detect the relevant Odoo model based on user message"""
    print("🧠 FATAL DEBUG: detect_model() was called!")
    logging.debug(f"[detect_model] Message: {message}")
    
    # Convert message to lowercase for better matching
    message_lower = message.lower()

    model_scores = {}

    # Score-based detection
    for model, keywords in MODEL_MAPPINGS.items():
        score = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in message_lower:
                # Check for word boundaries
                if f" {keyword_lower} " in f" {message_lower} " or message_lower.startswith(keyword_lower) or message_lower.endswith(keyword_lower):
                    score += 3  # Strong match
                else:
                    score += 1  # Partial match
        if score > 0:
            model_scores[model] = score

    if model_scores:
        best_model = max(model_scores, key=model_scores.get)
        logging.info(f"[detect_model] Best match: {best_model}")
        return best_model

    # Enhanced fallback patterns with case-insensitive matching
    fallback_patterns = [
        r'\b(update|delete|edit|change|modify|mark)\s+id\s+\d+\b',
        r'\b(update|delete|edit|change|modify|mark)\s+record\s+\d+\b',
        r'\bupdate\s+\d+\b',
        r'\bdelete\s+\d+\b',
        r'\bedit\s+\d+\b',
        r'\bchange\s+\d+\b',
        r'\bmark\s+\d+\s+as\b',
        r'\bupdate\s+\d+\s+to\b',
        r'\bchange\s+\d+\s+to\b',
        r'\blead\s+no\s+\d+\b',  # Added for "Lead no 2500576"
        r'\bquotation\b',        # Added for quotation detection
        r'\bstatus\s+of\s+lead\b' # Added for status queries
    ]

    for pattern in fallback_patterns:
        if re.search(pattern, message_lower, re.IGNORECASE):
            logging.info("[detect_model] Fallback triggered: defaulting to 'crm.lead'")
            return 'crm.lead'

    logging.warning("[detect_model] No model match found.")
    return None
def parse_update_command(message):
    """
    Enhanced parser for update commands with better pattern matching
    """
    # Pattern 1: update lead 33 as won
    pattern1 = r"update\s+(?:lead\s+)?(?:id\s+)?(\d+)\s+(?:as|to)\s+(won|lost|new|qualified)"
    match1 = re.search(pattern1, message, re.IGNORECASE)
    
    # Pattern 2: mark id 49 as won lead
    pattern2 = r"mark\s+(?:id\s+)?(\d+)\s+as\s+(?:a\s+)?(won|lost|new|qualified)\s*(?:lead)?"
    match2 = re.search(pattern2, message, re.IGNORECASE)
    
    # Pattern 3: update 49 id as won lead
    pattern3 = r"update\s+(\d+)\s+id\s+as\s+(won|lost|new|qualified)"
    match3 = re.search(pattern3, message, re.IGNORECASE)
    
    if match1:
        return match1.group(1), match1.group(2).lower()
    elif match2:
        return match2.group(1), match2.group(2).lower()
    elif match3:
        return match3.group(1), match3.group(2).lower()
    
    return None, None
# Enhanced extract_filters function for better date and status filtering
def extract_filters(message, model):
    """Enhanced filter extraction with better date and status handling"""
    filters = []
    message = message.lower()
    
    # Determine if we're looking for status-based leads
    is_won_query = (re.search(r'\bwon\s+leads?\b', message) or 
                   re.search(r'\bleads?\s+won\b', message) or
                   re.search(r'\bshow.*won\b', message))
    
    is_lost_query = (re.search(r'\blost\s+leads?\b', message) or 
                    re.search(r'\bleads?\s+lost\b', message))
    
    is_new_query = (re.search(r'\bnew\s+leads?\b', message) or 
                   re.search(r'\bleads?\s+new\b', message))
    
    # Status filters for leads
    if model == 'crm.lead':
        if is_won_query:
            try:
                stage_search = odoo.execute_kw(
                    ODOO_DB, uid, ODOO_PASSWORD,
                    'crm.stage', 'search_read',
                    [['name', '=', 'Won']],
                    {'fields': ['id'], 'limit': 1}
                )
                if stage_search:
                    filters.append(('stage_id', '=', stage_search[0]['id']))
            except Exception as e:
                logging.error(f"Error finding Won stage: {e}")
                
        elif is_lost_query:
            try:
                stage_search = odoo.execute_kw(
                    ODOO_DB, uid, ODOO_PASSWORD,
                    'crm.stage', 'search_read',
                    [['name', '=', 'Lost']],
                    {'fields': ['id'], 'limit': 1}
                )
                if stage_search:
                    filters.append(('stage_id', '=', stage_search[0]['id']))
            except Exception as e:
                logging.error(f"Error finding Lost stage: {e}")
                
        elif is_new_query:
            try:
                stage_search = odoo.execute_kw(
                    ODOO_DB, uid, ODOO_PASSWORD,
                    'crm.stage', 'search_read',
                    [['name', '=', 'New']],
                    {'fields': ['id'], 'limit': 1}
                )
                if stage_search:
                    filters.append(('stage_id', '=', stage_search[0]['id']))
            except Exception as e:
                logging.error(f"Error finding New stage: {e}")
    
    # Enhanced date filters - use appropriate date field based on query type
    today = datetime.now()
    
    # Determine which date field to use based on the query type
    if is_won_query or is_lost_query:
        # For won/lost leads, use the date when they were moved to that stage
        date_field = 'date_closed'  # or 'write_date' if date_closed is not available
    else:
        # For new leads or general queries, use creation date
        date_field = 'create_date'
    
    if re.search(r'\btoday\b', message):
        today_str = today.strftime('%Y-%m-%d')
        filters.append((date_field, '>=', today_str + ' 00:00:00'))
        filters.append((date_field, '<=', today_str + ' 23:59:59'))
        
    elif re.search(r'\byesterday\b', message):
        yesterday = today - timedelta(days=1)
        yesterday_str = yesterday.strftime('%Y-%m-%d')
        filters.append((date_field, '>=', yesterday_str + ' 00:00:00'))
        filters.append((date_field, '<=', yesterday_str + ' 23:59:59'))
        
    elif re.search(r'\bthis\s+week\b', message):
        week_start = today - timedelta(days=today.weekday())
        filters.append((date_field, '>=', week_start.strftime('%Y-%m-%d 00:00:00')))
        
    elif re.search(r'\bthis\s+month\b', message):
        month_start = today.replace(day=1)
        filters.append((date_field, '>=', month_start.strftime('%Y-%m-%d 00:00:00')))
        
    elif re.search(r'\blast\s+week\b', message):
        week_start = today - timedelta(days=today.weekday() + 7)
        week_end = week_start + timedelta(days=6)
        filters.append((date_field, '>=', week_start.strftime('%Y-%m-%d 00:00:00')))
        filters.append((date_field, '<=', week_end.strftime('%Y-%m-%d 23:59:59')))
        
    elif re.search(r'\blast\s+month\b', message):
        last_month = today.replace(day=1) - timedelta(days=1)
        month_start = last_month.replace(day=1)
        filters.append((date_field, '>=', month_start.strftime('%Y-%m-%d 00:00:00')))
        filters.append((date_field, '<', today.replace(day=1).strftime('%Y-%m-%d 00:00:00')))
    
    # Amount filters
    amount_match = re.search(r'amount\s*[><=]\s*(\d+)', message)
    if amount_match:
        amount = int(amount_match.group(1))
        if '>' in message:
            filters.append(('amount_total', '>', amount))
        elif '<' in message:
            filters.append(('amount_total', '<', amount))
        elif '=' in message:
            filters.append(('amount_total', '=', amount))
    
    # Name filters
    name_match = re.search(r'name.*"([^"]+)"', message)
    if name_match:
        filters.append(('name', 'ilike', name_match.group(1)))
    
    return filters
def extract_fields(message, model):
    """Extract field list from message"""
    # Default fields for common models
    default_fields = {
        'crm.lead': ['name', 'partner_id', 'email_from', 'phone', 'expected_revenue', 'stage_id', 'create_date'],
        'res.partner': ['name', 'email', 'phone', 'city', 'country_id', 'is_company', 'create_date'],
        'sale.order': ['name', 'partner_id', 'date_order', 'amount_total', 'state', 'create_date'],
        'account.move': ['name', 'partner_id', 'invoice_date', 'amount_total', 'state', 'create_date'],
        'hr.employee': ['name', 'work_email', 'mobile_phone', 'department_id', 'job_id', 'create_date'],
        'project.task': ['name', 'project_id', 'user_id', 'deadline', 'stage_id', 'create_date'],
        'product.product': ['name', 'list_price', 'standard_price', 'categ_id', 'active', 'create_date']
    }
    
    if model in default_fields:
        return default_fields[model]
    
    return ['name', 'create_date']

def extract_limit(message):
    """Extract record limit from message"""
    # Look for numbers in the message
    numbers = re.findall(r'\b(\d+)\b', message)
    if numbers:
        # Take the first reasonable number (between 1 and 1000)
        for num in numbers:
            num = int(num)
            if 1 <= num <= 1000:
                return num
    
    # Default limits based on keywords
    if "all" in message.lower():
        return None
    elif "top" in message.lower() or "first" in message.lower():
        return 10
    
    return 50  # Default limit
def extract_record_id(message):
    """Smart record ID extractor that avoids phone/amount confusion"""
    message = message.lower()

    # 🚫 Skip common non-ID patterns (phone, amount, etc.)
    skip_patterns = [
        r'phone\s+number\s*\d+',
        r'mobile\s+number\s*\d+',
        r'amount\s*\d+',
        r'number\s*\d{7,}',  # 7+ digits = likely phone number
    ]
    for pattern in skip_patterns:
        if re.search(pattern, message):
            return None

    # ✅ Valid ID patterns
    id_patterns = [
        r'\blead\s+(\d+)\b',
        r'\brecord\s+(\d+)\b',
        r'id\s*:?\s*(\d+)',
        r'#(\d+)',
        r'\bupdate\s+id\s*(\d+)\b',
        r'\bmark\s+(?:id\s+)?(\d+)\b',
        r'\bupdate\s+(?:lead\s+)?(?:id\s+)?(\d+)\b',
        r'\bupdate\s+(\d+)\s+id\b',
        r'\bdelete\s+(?:lead\s+)?(?:id\s+)?(\d+)\b',
        r'\bchange\s+(?:lead\s+)?(?:id\s+)?(\d+)\b',
        r'\bmodify\s+(?:lead\s+)?(?:id\s+)?(\d+)\b',
    ]

    for pattern in id_patterns:
        match = re.search(pattern, message)
        if match:
            return int(match.group(1))

    return None  # Nothing found


def find_record_by_fuzzy_match(model, search_value, search_fields=['name', 'display_name'], threshold=80):
    """Find records using fuzzy matching instead of exact ID"""
    try:
        # Get all records with basic fields
        all_records = odoo.execute_kw(
            ODOO_DB, uid, ODOO_PASSWORD,
            model, 'search_read',
            [[]],
            {'fields': search_fields + ['id'], 'limit': 200}
        )
        
        best_matches = []
        for record in all_records:
            for field in search_fields:
                if field in record and record[field]:
                    # Handle Many2one fields
                    field_value = record[field]
                    if isinstance(field_value, list):
                        field_value = field_value[1]
                    
                    score = fuzz.ratio(search_value.lower(), str(field_value).lower())
                    if score >= threshold:
                        best_matches.append((record['id'], field_value, score))
        
        # Sort by score and return best match
        if best_matches:
            best_matches.sort(key=lambda x: x[2], reverse=True)
            return best_matches[0][0]  # Return ID of best match
        
        return None
        
    except Exception as e:
        logging.error(f"Error in fuzzy matching: {e}")
        return None

# Replace your existing parse_field_values function with this enhanced version
def parse_field_values(message, model):
    """Enhanced field parser to handle flexible natural language"""
    values = {}
    message_lower = message.lower()
    
    # FIXED: Handle status update commands (won/lost/qualified/new)
    status_patterns = [
        r'(?:update|mark|change|set)\s+(?:lead\s+)?(?:.*?)\s+(?:as|to)\s+(won|lost|qualified|new)',
        r'(?:update|mark|change|set)\s+(?:.*?)\s+(?:as|to)\s+(?:a\s+)?(won|lost|qualified|new)(?:\s+lead)?',
        r'(?:mark|set)\s+(?:.*?)\s+(?:as|to)\s+(won|lost|qualified|new)',
        r'(?:as|to)\s+(?:a\s+)?(won|lost|qualified|new)(?:\s+lead)?'
    ]
    
    for pattern in status_patterns:
        match = re.search(pattern, message_lower)
        if match:
            status = match.group(1).lower()
            stage_mapping = {
                'won': 'Won',
                'lost': 'Lost', 
                'new': 'New',
                'qualified': 'Qualified'
            }
            
            if status in stage_mapping:
                try:
                    # Import the necessary modules at the top of your file
                    # from your_odoo_connection import odoo, ODOO_DB, uid, ODOO_PASSWORD
                    
                    stage_search = odoo.execute_kw(
                        ODOO_DB, uid, ODOO_PASSWORD,
                        'crm.stage', 'search_read',
                        [[['name', '=', stage_mapping[status]]]],
                        {'fields': ['id'], 'limit': 1}
                    )
                    if stage_search:
                        values['stage_id'] = stage_search[0]['id']
                        break  # Exit the loop once we find a match
                except Exception as e:
                    logging.error(f"Error finding stage: {e}")
            break  # Exit the outer loop once we find a status pattern

    # Extract Email - FIXED PATTERNS
    email_patterns = [
        r'email\s+(?:from\s+)?(?:with\s+)?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        r'with\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\s+email',
        r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\s+email',
        r'email\s*[:=]?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        r'email is\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        r'with email\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    ]
    
    for pattern in email_patterns:
        match = re.search(pattern, message_lower)
        if match:
            if model == 'crm.lead':
                values['email_from'] = match.group(1)
            else:
                values['email'] = match.group(1)
            break

    # Extract Phone - ENHANCED PATTERNS
    phone_patterns = [
        r'phone\s+(?:number\s+)?(?:with\s+)?(\d{6,})',
        r'with\s+(\d{8,})\s+phone',
        r'(\d{8,})\s+phone',
        r'phone\s*[:=]?\s*(\d{6,})',
        r'phone number\s*[:=]?\s*(\d{6,})',
        r'phone is\s*(\d{6,})',
        r'by\s+(\d{8,})\s+phone'
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, message_lower)
        if match:
            values['phone'] = match.group(1).strip()
            break

    # Extract Name - ENHANCED PATTERNS
    name_patterns = [
        r'(?:create|add|new)\s+(?:lead|customer|partner|sales order|opportunity)\s+(?:for|named)?\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        r'(?:create|add|new)\s+(?:lead|customer|partner|sales order|opportunity)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s+with|\s+for|\s+phone|\s+email|\s+rupees|$)',
        r'(?:lead|customer|partner|sales order|opportunity)\s+(?:for|named)?\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s+with|\s+for|\s+phone|\s+email|\s+rupees|$)',
        r'name\s*[:=]?\s*["\']?([a-zA-Z]+(?:\s+[a-zA-Z]+)*?)["\']?(?:\s+with|\s+for|\s+phone|\s+email|\s+rupees|$)',
        r'called\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s+with|\s+for|\s+phone|\s+email|\s+rupees|$)',
        r'update\s+lead\s+name\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s+|$)'
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, message_lower)
        if match:
            name = match.group(1).strip()
            if len(name) > 1 and name not in ['with', 'for', 'phone', 'email', 'rupees', 'rs', 'and', 'the', 'a', 'an', 'won', 'lost', 'qualified', 'new']:
                values['name'] = name
                break

    # Extract Amount - ENHANCED PATTERNS
    amount_patterns = [
        r'(?:amount|for|with|rupees|rs\.?|₹)\s*[:=]?\s*(\d+(?:\.\d+)?)',
        r'amount\s*is\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:rupees|rs\.?|₹)',
        r'with\s+(\d+(?:\.\d+)?)\s*(?:rupees|rs\.?|₹)?(?:\s+rupees)?',
        r'price\s+(?:with\s+)?(\d+(?:\.\d+)?)',
        r'revenue\s+(?:with\s+)?(\d+(?:\.\d+)?)'
    ]
    
    for pattern in amount_patterns:
        match = re.search(pattern, message_lower)
        if match:
            amount = float(match.group(1))
            if model == 'crm.lead':
                values['expected_revenue'] = amount
            elif model in ['sale.order', 'account.move']:
                values['amount_total'] = amount
            else:
                values['amount'] = amount
            break

    # Extract Priority
    if 'high priority' in message_lower or 'urgent' in message_lower:
        values['priority'] = '3'
    elif 'medium priority' in message_lower:
        values['priority'] = '2'
    elif 'low priority' in message_lower:
        values['priority'] = '1'

    # Extract Description
    desc_match = re.search(r'description\s*[:=]?\s*"([^"]+)"', message_lower)
    if desc_match:
        values['description'] = desc_match.group(1)

    return values

# Enhanced extract_record_id function
def extract_record_id(message):
    """Extract record ID or name from the message with patterns and fallback to session"""
    import re
    from flask import session

    # 1. Try numeric patterns (ID-based)
    id_patterns = [
        r'\blead\s+(\d+)\b',
        r'\brecord\s+(\d+)\b',
        r'id\s*:?\s*(\d+)',
        r'#(\d+)',
        r'number\s*:?\s*(\d+)',
        r'\bupdate\s+(\d+)\b',
        r'\bmark\s+(?:id\s+)?(\d+)\b',
        r'\bupdate\s+(?:lead\s+)?(?:id\s+)?(\d+)\b',
        r'\bupdate\s+(\d+)\s+id\b'
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, message.lower())
        if match:
            return int(match.group(1))

    # 2. Try name-based patterns (fuzzy match)
    name_patterns = [
        r'update\s+(?:lead\s+)?(?:customer\s+)?(.+?)\s+(?:to|as|with)',
        r'mark\s+(?:lead\s+)?(?:customer\s+)?(.+?)\s+as',
        r'delete\s+(?:lead\s+)?(?:customer\s+)?(.+?)(?:\s|$)',
        r'(?:lead|customer|partner)\s+(.+?)\s+(?:to|as|with|$)'
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, message.lower())
        if match:
            search_name = match.group(1).strip()
            if search_name.lower() not in ['id', 'with', 'to', 'as', 'the', 'a', 'an']:
                return search_name  # name-based reference

    # 3. Fallback: session memory
    return session.get("last_record")



def resolve_relational_fields(values, model):
    """Resolve relational field values to IDs"""
    relational_mappings = {
        'partner_id': 'res.partner',
        'user_id': 'res.users',
        'company_id': 'res.company',
        'product_id': 'product.product',
        'project_id': 'project.project',
        'team_id': 'crm.team',
        'department_id': 'hr.department',
        'stage_id': {
            'crm.lead': 'crm.stage',
            'project.task': 'project.task.type'
        }
    }
    
    for field, target_model in relational_mappings.items():
        if field in values and isinstance(values[field], str):
            try:
                # Handle nested mappings
                if isinstance(target_model, dict):
                    target_model = target_model.get(model, 'res.partner')
                
                # Search for the record
                search_result = odoo.execute_kw(
                    ODOO_DB, uid, ODOO_PASSWORD,
                    target_model, 'search_read',
                    [['|', ('name', 'ilike', values[field]), ('display_name', 'ilike', values[field])]],
                    {'fields': ['id', 'name'], 'limit': 1}
                )
                
                if search_result:
                    values[field] = search_result[0]['id']
                else:
                    # Try to create if it's a simple model
                    if target_model in ['res.partner', 'crm.team']:
                        new_id = odoo.execute_kw(
                            ODOO_DB, uid, ODOO_PASSWORD,
                            target_model, 'create',
                            [{'name': values[field]}]
                        )
                        values[field] = new_id
                    else:
                        # Remove the field if we can't resolve it
                        del values[field]
            except Exception as e:
                logging.error(f"Error resolving {field}: {e}")
                # Remove the field if we can't resolve it
                if field in values:
                    del values[field]
    
    return values

def generate_html_table(data, fields, title="Results"):
    """Generate HTML table from data"""
    if not data:
        return f"<div class='no-results'><h3>{title}</h3><p>No records found.</p></div>"
    
    # Ensure 'id' is always shown first
    if 'id' not in fields:
        fields = ['id'] + fields

    html = f"<div class='results-table'><h3>{title} ({len(data)} records)</h3>"
    html += "<table border='1' cellpadding='8' cellspacing='0' style='border-collapse: collapse; width: 100%;'>"
    
    # Headers
    html += "<thead><tr style='background-color: #f0f0f0;'>"
    for field in fields:
        display_name = field.replace('_', ' ').title()
        html += f"<th style='padding: 8px; border: 1px solid #ddd;'>{display_name}</th>"
    html += "</tr></thead>"
    
    # Data rows
    html += "<tbody>"
    for i, record in enumerate(data):
        bg_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        html += f"<tr style='background-color: {bg_color};'>"
        
        for field in fields:
            value = record.get(field, 'N/A')
            
            # Handle Many2One fields
            if isinstance(value, list) and len(value) > 1:
                value = value[1]
            elif isinstance(value, list) and len(value) == 1:
                value = value[0]
            
            # Format currency fields
            if field in ['amount_total', 'expected_revenue', 'list_price', 'standard_price'] and isinstance(value, (int, float)):
                value = f"${value:,.2f}"
            
            # Format date fields
            if field.endswith('_date') and value != 'N/A':
                try:
                    if isinstance(value, str):
                        date_obj = datetime.strptime(value.split(' ')[0], '%Y-%m-%d')
                        value = date_obj.strftime('%Y-%m-%d')
                except:
                    pass

            # Highlight ID
            if field == 'id':
                value = f"<b>{value}</b>"
            
            html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{value}</td>"
        
        html += "</tr>"
    
    html += "</tbody></table></div>"
    return html

def execute_odoo_operation(intent, model, filters=None, fields=None, values=None, record_id=None, limit=None):
    """Execute Odoo operation based on intent with enhanced error handling"""
    try:
        if intent == "create":
            if not values:
                return "<p>❌ No values provided for creation</p>"
            
            # Resolve relational fields
            values = resolve_relational_fields(values, model)
            
            result = odoo.execute_kw(
                ODOO_DB, uid, ODOO_PASSWORD,
                model, 'create', [values]
            )
            
            # Store the created record ID in session
            session["last_record"] = result
            
            return f"<div class='success'>✅ Successfully created {model} record with ID: {result}</div>"
        
        elif intent == "read":
            kwargs = {}
            if fields:
                kwargs['fields'] = fields
            if limit:
                kwargs['limit'] = limit
            
            if not filters:
                filters = []
            
            result = odoo.execute_kw(
                ODOO_DB, uid, ODOO_PASSWORD,
                model, 'search_read', [filters], kwargs
            )
            
            if not fields:
                fields = ['name', 'create_date']
            
            return generate_html_table(result, fields, f"{model.replace('_', ' ').title()} Records")
        
        elif intent == "update":
            if not record_id:
                return "<p>❌ Record ID required for update operation</p>"
            
            if not values:
                return "<p>❌ No values provided for update</p>"
            
            # Resolve relational fields
            values = resolve_relational_fields(values, model)
            
            # Verify record exists before update
            existing_record = odoo.execute_kw(
                ODOO_DB, uid, ODOO_PASSWORD,
                model, 'search_read',
                [[('id', '=', record_id)]],
                {'fields': ['id', 'name'], 'limit': 1}
            )
            
            if not existing_record:
                return f"<div class='error'>❌ Record with ID {record_id} not found in {model}</div>"
            
            result = odoo.execute_kw(
                ODOO_DB, uid, ODOO_PASSWORD,
                model, 'write', [[record_id], values]
            )
            
            if result:
                # Update session with the record ID
                session["last_record"] = record_id
                return f"<div class='success'>✅ Successfully updated {model} record ID: {record_id}</div>"
            else:
                return f"<div class='error'>❌ Failed to update {model} record ID: {record_id}</div>"
        
        elif intent == "delete":
            if not record_id:
                return "<p>❌ Record ID required for delete operation</p>"
            
            # Verify record exists before deletion
            existing_record = odoo.execute_kw(
                ODOO_DB, uid, ODOO_PASSWORD,
                model, 'search_read',
                [[('id', '=', record_id)]],
                {'fields': ['id', 'name'], 'limit': 1}
            )
            
            if not existing_record:
                return f"<div class='error'>❌ Record with ID {record_id} not found in {model}</div>"
            
            result = odoo.execute_kw(
                ODOO_DB, uid, ODOO_PASSWORD,
                model, 'unlink', [[record_id]]
            )
            
            if result:
                # Clear session record since it's deleted
                if session.get("last_record") == record_id:
                    session.pop("last_record", None)
                return f"<div class='success'>✅ Successfully deleted {model} record ID: {record_id}</div>"
            else:
                return f"<div class='error'>❌ Failed to delete {model} record ID: {record_id}</div>"
        
        elif intent == "count":
            if not filters:
                filters = []

            # Get the count
            count_result = odoo.execute_kw(
                ODOO_DB, uid, ODOO_PASSWORD,
                model, 'search_count', [filters]
            )

            # Get sample records with the same filters
            result = odoo.execute_kw(
                ODOO_DB, uid, ODOO_PASSWORD,
                model, 'search_read', [filters],
                {'fields': fields, 'limit': min(count_result, 20)}  # Show up to 20 records
            )

            # Generate appropriate title based on filters
            title = f"{model.replace('_', ' ').title()} Records"
            if filters:
                for filter_item in filters:
                    if isinstance(filter_item, tuple) and len(filter_item) == 3:
                        field, operator, value = filter_item
                        if field == 'stage_id' and operator == '=':
                            # Get stage name
                            try:
                                stage_info = odoo.execute_kw(
                                    ODOO_DB, uid, ODOO_PASSWORD,
                                    'crm.stage', 'read', [value], {'fields': ['name']}
                                )
                                if stage_info:
                                    title = f"{stage_info[0]['name']} {model.replace('_', ' ').title()} Records"
                            except:
                                pass

            records_html = generate_html_table(result, fields, title)

            return f"""
                <div class='info'>📊 Total {model.replace('_', ' ').title()} records: {count_result}</div>
                {records_html}
            """

        else:
            return f"<p>❌ Unsupported operation: {intent}</p>"
    
    except Exception as e:
        logging.error(f"Error executing {intent} on {model}: {e}")
        return f"<div class='error'>❌ Error: {str(e)}</div>"


def execute_formula(formula, context, models=None, uid=None, password=None, db=None):
    try:
        # Replace "'Won'" with actual stage ID if needed
        if 'stage_id' in formula and "'Won'" in formula and models:
            stage = models.execute_kw(db, uid, password,
                'crm.stage', 'search_read',
                [[['name', '=', 'Won']]],
                {'fields': ['id'], 'limit': 1})
            if stage:
                formula = formula.replace("'Won'", str(stage[0]['id']))
        result = eval(formula, {"__builtins__": {}}, context)
        return result
    except Exception as e:
        return f"Formula execution error: {str(e)}"
def find_record_by_fuzzy_match(model, search_value, search_fields=['name', 'display_name'], threshold=80):
    """Find records using fuzzy matching instead of exact ID"""
    try:
        # Get all records with basic fields
        all_records = odoo.execute_kw(
            ODOO_DB, uid, ODOO_PASSWORD,
            model, 'search_read',
            [[]],
            {'fields': search_fields + ['id'], 'limit': 200}
        )
        
        best_matches = []
        for record in all_records:
            for field in search_fields:
                if field in record and record[field]:
                    # Handle Many2one fields
                    field_value = record[field]
                    if isinstance(field_value, list):
                        field_value = field_value[1]
                    
                    score = fuzz.ratio(search_value.lower(), str(field_value).lower())
                    if score >= threshold:
                        best_matches.append((record['id'], field_value, score))
        
        # Sort by score and return best match
        if best_matches:
            best_matches.sort(key=lambda x: x[2], reverse=True)
            return best_matches[0][0]  # Return ID of best match
        
        return None
        
    except Exception as e:
        logging.error(f"Error in fuzzy matching: {e}")
        return None



def translate_to_english(text):
    try:
        result = subprocess.run(
            [
                "../venv_googletrans/bin/python",
                "translator/translate_text.py",
                text
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.stderr:
            print("Subprocess stderr:", result.stderr)

        if not result.stdout.strip():
            print("⚠️ No stdout from subprocess. Returning original text.")
            return text

        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError as json_err:
            print("⚠️ JSON Decode Error:", json_err)
            print("Raw output was:", result.stdout)
            return text

        if isinstance(output, dict) and output.get("success"):
            return output.get("result", text)
        else:
            print("❌ Translation failed or malformed output:", output)
            return text

    except Exception as e:
        print("Subprocess failed:", e)
        return text


def get_today_won_leads(self):
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Skip stage lookup and use probability instead
        domain = [
            ('write_date', '>=', today), 
            ('probability', '=', 100),
            ('active', '=', True)
        ]
        return self._execute('crm.lead', 'search_read', domain, ['name', 'write_date'])
        
    except Exception as e:
        logging.error(f"Error in won leads: {e}")
        # Final fallback - just today's leads
        domain = [('create_date', '>=', today)]
        return self._execute('crm.lead', 'search_read', domain, ['name', 'create_date'])



def classify_intent(message):
    """Classify the intent of the message with better pattern matching"""
    message_lower = message.lower()
    
    # UPDATE operations - check these first
    update_patterns = [
        r'(?:update|mark|change|set)\s+(?:lead\s+)?(?:.*?)\s+(?:as|to)\s+(?:a\s+)?(?:won|lost|qualified|new)',
        r'(?:mark|set)\s+(?:.*?)\s+(?:as|to)\s+(?:a\s+)?(?:won|lost|qualified|new)(?:\s+lead)?',
        r'update\s+(?:lead\s+)?(?:.*?)\s+(?:with|to)',
        r'mark\s+(?:.*?)\s+as',
        r'change\s+(?:.*?)\s+to',
        r'set\s+(?:.*?)\s+as'
    ]
    
    for pattern in update_patterns:
        if re.search(pattern, message_lower):
            return "update"
    
    # CREATE operations
    create_patterns = [
        r'(?:create|add|new)\s+(?:lead|customer|partner|sales order|opportunity)',
        r'(?:lead|customer|partner|sales order|opportunity)\s+for\s+[a-zA-Z]',
        r'create\s+[a-zA-Z]'
    ]
    
    for pattern in create_patterns:
        if re.search(pattern, message_lower):
            return "create"
    
    # DELETE operations
    delete_patterns = [
        r'(?:delete|remove)\s+(?:lead\s+)?(?:.*?)',
        r'(?:lead\s+)?(?:.*?)\s+delete'
    ]
    
    for pattern in delete_patterns:
        if re.search(pattern, message_lower):
            return "delete"
    
    # COUNT operations
    count_patterns = [
        r'(?:count|total|how many)\s+(?:leads|customers|partners)',
        r'(?:leads|customers|partners)\s+(?:count|total)',
        r'show\s+(?:all\s+)?(?:leads|customers|partners)'
    ]
    
    for pattern in count_patterns:
        if re.search(pattern, message_lower):
            return "count"
    
    # READ operations (default for listing)
    read_patterns = [
        r'(?:show|list|get|find|search)\s+(?:leads|customers|partners)',
        r'(?:leads|customers|partners)\s+(?:show|list)',
        r'(?:all\s+)?(?:leads|customers|partners)'
    ]
    
    for pattern in read_patterns:
        if re.search(pattern, message_lower):
            return "read"
    
    # Default fallback
    return "read"


def resolve_record_id(message, model):
    """Enhanced record ID resolver with better name matching"""
    message_lower = message.lower()
    
    # 1. Try to extract explicit ID from message
    explicit_id = extract_record_id(message)
    if explicit_id and isinstance(explicit_id, int):
        # Verify the ID exists in the model
        found_id = find_record_by_name_or_id(model, explicit_id)
        if found_id:
            return found_id
    
    # 2. Try to extract name-based reference for UPDATE operations
    if any(word in message_lower for word in ['update', 'mark', 'change', 'set']):
        name_patterns = [
            r'(?:update|mark|change|set)\s+(?:lead\s+)?(.+?)\s+(?:as|to|with)',
            r'(?:update|mark|change|set)\s+(.+?)\s+(?:as|to|with)',
            r'mark\s+(.+?)\s+(?:as|to)',
            r'update\s+(.+?)\s+(?:as|to|with)',
            r'change\s+(.+?)\s+(?:to|as)',
            r'set\s+(.+?)\s+(?:as|to)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message_lower)
            if match:
                search_name = match.group(1).strip()
                # Clean up the name - remove common words
                search_name = re.sub(r'\b(lead|customer|partner|the|a|an)\b', '', search_name).strip()
                
                if search_name and len(search_name) > 1:
                    found_id = find_record_by_name_or_id(model, search_name)
                    if found_id:
                        return found_id
    
    # 3. Try general name patterns for other operations
    general_name_patterns = [
        r'(?:lead|customer|partner)\s+(.+?)\s+(?:to|as|with|$)',
        r'(?:delete|remove)\s+(?:lead\s+)?(.+?)(?:\s|$)',
        r'find\s+(.+?)(?:\s|$)'
    ]
    
    for pattern in general_name_patterns:
        match = re.search(pattern, message_lower)
        if match:
            search_name = match.group(1).strip()
            # Clean up the name
            search_name = re.sub(r'\b(lead|customer|partner|the|a|an|as|to|with)\b', '', search_name).strip()
            
            if search_name and len(search_name) > 1:
                found_id = find_record_by_name_or_id(model, search_name)
                if found_id:
                    return found_id
    
    # 4. Fallback to session memory
    session_id = session.get("last_record")
    if session_id:
        # Verify session ID still exists
        found_id = find_record_by_name_or_id(model, session_id)
        if found_id:
            return found_id
    
    return None


def find_record_by_name_or_id(model, identifier):
    """Find record by ID or name with enhanced fuzzy matching"""
    if not identifier:
        return None
        
    # If it's a number, try ID first
    if isinstance(identifier, int) or str(identifier).isdigit():
        try:
            record_id = int(identifier)
            # Check if record exists
            result = odoo.execute_kw(
                ODOO_DB, uid, ODOO_PASSWORD,
                model, 'search_read',
                [[('id', '=', record_id)]],
                {'fields': ['id', 'name'], 'limit': 1}
            )
            if result:
                return record_id
        except:
            pass
    
    # Try fuzzy name matching
    if isinstance(identifier, str):
        return find_record_by_fuzzy_match(model, identifier)
    
    return None



def extract_quotation_values(message):
    """Extract quotation-related values from message"""
    values = {}
    
    # Extract lead number for quotation
    lead_match = re.search(r'\blead(?:\s+no\.?|\s*#)?\s*(\d{4,})\b', message, re.IGNORECASE)

    if lead_match:
        values['lead_id'] = lead_match.group(1)
    
    # Extract product name
    product_match = re.search(r'product\s+([A-Za-z0-9\s]+?)(?:\s+with|\s+qty|$)', message, re.IGNORECASE)
    if product_match:
        values['product_name'] = product_match.group(1).strip()
    
    # Extract quantity
    qty_match = re.search(r'(\d+)\s+qty', message, re.IGNORECASE)
    if qty_match:
        values['quantity'] = int(qty_match.group(1))
    
    # Extract customer name
    customer_match = re.search(r'(?:send\s+it\s+to|to|for)\s+([A-Za-z\s]+)', message, re.IGNORECASE)
    if customer_match:
        values['customer_name'] = customer_match.group(1).strip()
    
    return values

def extract_lead_values(message):
    """Extract lead-related values from message"""
    values = {}
    
    # Extract lead number/ID
    lead_match = re.search(r'lead\s+no\s+(\d+)', message, re.IGNORECASE)
    if lead_match:
        values['lead_id'] = lead_match.group(1)
    
    # Extract customer/contact name
    customer_match = re.search(r'(?:to|for|customer|client)\s+([A-Za-z\s]+)', message, re.IGNORECASE)
    if customer_match:
        values['customer_name'] = customer_match.group(1).strip()
    
    return values


def extract_values_by_model(message, model):
    """Extract values based on detected model"""
    if model == 'crm.lead':
        if 'quotation' in message.lower():
            return extract_quotation_values(message)
        else:
            return extract_lead_values(message)
    elif model == 'sale.order':
        return extract_quotation_values(message)
    
    return {}


def create_lead(values):
    """Create a new lead in Odoo"""
    try:
        # Check if required values exist
        if not values:
            return "❌ No values provided for lead creation"
        
        # Prepare lead data
        lead_data = {}
        
        # Handle customer name
        if 'customer_name' in values:
            lead_data['name'] = values['customer_name']
            lead_data['partner_name'] = values['customer_name']
        
        # Handle lead ID (if updating existing lead)
        if 'lead_id' in values:
            lead_data['id'] = values['lead_id']
        
        # Set default values if not provided
        if 'name' not in lead_data:
            lead_data['name'] = 'New Lead'
        
        # Debug output
        print(f"🔍 Creating lead with data: {lead_data}")
        
        # Connect to Odoo and create lead
        if odoo:  # Make sure odoo connection exists
            try:
                if 'id' in lead_data:
                    # Update existing lead
                    lead_id = lead_data.pop('id')
                    result = odoo.env['crm.lead'].browse(int(lead_id)).write(lead_data)
                    return f"✅ Updated Lead {lead_id}"
                else:
                    # Create new lead
                    result = odoo.env['crm.lead'].create(lead_data)
                    return f"✅ Created Lead {result.id}"
            except Exception as e:
                return f"❌ Odoo error: {str(e)}"
        else:
            return "❌ Odoo connection not available"
            
    except Exception as e:
        print(f"❌ Error in create_lead: {str(e)}")
        return f"❌ Error creating lead: {str(e)}"

def create_quotation_from_lead(values):
    """Create quotation from existing lead - FIXED VERSION"""
    try:
        print(f"🔧 Creating quotation with values: {values}")
        
        lead_id = values.get('lead_id')
        if not lead_id:
            return "❌ Lead ID is required for quotation"
        
        # Check Odoo connection
        if not globals().get('odoo'):
            return "❌ Odoo connection not available"
        
        try:
            # Method 1: Using odoo.env (if you have ORM access)
            if hasattr(odoo, 'env'):
                lead = odoo.env['crm.lead'].browse(int(lead_id))
                if not lead.exists():
                    return f"❌ Lead {lead_id} not found"
                
                # Create quotation data
                quotation_data = {
                    'opportunity_id': lead.id,
                    'order_line': []
                }
                
                # Set partner
                if lead.partner_id:
                    quotation_data['partner_id'] = lead.partner_id.id
                elif values.get('customer_name'):
                    # Find or create partner
                    partner = odoo.env['res.partner'].search([('name', '=', values['customer_name'])], limit=1)
                    if not partner:
                        partner = odoo.env['res.partner'].create({'name': values['customer_name']})
                    quotation_data['partner_id'] = partner.id
                
                # Add product if specified
                if values.get('product_name'):
                    product_name = values['product_name']
                    quantity = values.get('quantity', 1)
                    
                    # Find or create product
                    product = odoo.env['product.product'].search([('name', 'ilike', product_name)], limit=1)
                    if not product:
                        product = odoo.env['product.product'].create({
                            'name': product_name,
                            'type': 'consu',
                            'list_price': 100.0
                        })
                    
                    quotation_data['order_line'].append((0, 0, {
                        'product_id': product.id,
                        'product_uom_qty': quantity,
                        'name': product.name,
                        'price_unit': product.list_price
                    }))
                
                # Create quotation
                quotation = odoo.env['sale.order'].create(quotation_data)
                return f"✅ Created Quotation {quotation.name} for Lead {lead_id}"
            
            # Method 2: Using execute_kw (if you have XML-RPC access)
            elif hasattr(odoo, 'execute_kw'):
                # Check if lead exists
                lead_exists = odoo.execute_kw(
                    ODOO_DB, uid, ODOO_PASSWORD,
                    'crm.lead', 'search_read',
                    [[('id', '=', int(lead_id))]],
                    {'fields': ['id', 'name', 'partner_id'], 'limit': 1}
                )
                
                if not lead_exists:
                    return f"❌ Lead {lead_id} not found"
                
                lead_data = lead_exists[0]
                
                # Prepare quotation data
                quotation_data = {
                    'opportunity_id': int(lead_id),
                    'order_line': []
                }
                
                # Set partner
                if lead_data.get('partner_id'):
                    quotation_data['partner_id'] = lead_data['partner_id'][0]
                elif values.get('customer_name'):
                    # Find or create partner
                    partner = odoo.execute_kw(
                        ODOO_DB, uid, ODOO_PASSWORD,
                        'res.partner', 'search',
                        [[('name', '=', values['customer_name'])]]
                    )
                    if not partner:
                        partner_id = odoo.execute_kw(
                            ODOO_DB, uid, ODOO_PASSWORD,
                            'res.partner', 'create',
                            [{'name': values['customer_name']}]
                        )
                        quotation_data['partner_id'] = partner_id
                    else:
                        quotation_data['partner_id'] = partner[0]
                
                # Add product if specified
                if values.get('product_name'):
                    product_name = values['product_name']
                    quantity = values.get('quantity', 1)
                    
                    # Find or create product
                    product = odoo.execute_kw(
                        ODOO_DB, uid, ODOO_PASSWORD,
                        'product.product', 'search',
                        [[('name', 'ilike', product_name)]]
                    )
                    
                    if not product:
                        product_id = odoo.execute_kw(
                            ODOO_DB, uid, ODOO_PASSWORD,
                            'product.product', 'create',
                            [{'name': product_name, 'type': 'consu', 'list_price': 100.0}]
                        )
                    else:
                        product_id = product[0]
                    
                    quotation_data['order_line'] = [(0, 0, {
                        'product_id': product_id,
                        'product_uom_qty': quantity,
                        'name': product_name,
                        'price_unit': 100.0
                    })]
                
                # Create quotation
                quotation_id = odoo.execute_kw(
                    ODOO_DB, uid, ODOO_PASSWORD,
                    'sale.order', 'create',
                    [quotation_data]
                )
                
                return f"✅ Created Quotation {quotation_id} for Lead {lead_id}"
            
            else:
                return "❌ Unknown Odoo connection type"
                
        except Exception as e:
            print(f"❌ Odoo error: {str(e)}")
            return f"❌ Odoo error: {str(e)}"
            
    except Exception as e:
        print(f"❌ Error in create_quotation_from_lead: {str(e)}")
        return f"❌ Error creating quotation: {str(e)}"

def handle_status_query(message, model):
    """Handle status queries"""
    try:
        lead_match = re.search(r'lead\s+no\s+(\d+)', message, re.IGNORECASE)
        if lead_match:
            lead_id = lead_match.group(1)
            
            if hasattr(odoo, 'env'):
                lead = odoo.env['crm.lead'].browse(int(lead_id))
                if lead.exists():
                    return f"✅ Lead {lead_id} Status: {lead.stage_id.name}, Partner: {lead.partner_name or 'Not Set'}, Revenue: ${lead.expected_revenue:,.2f}"
                else:
                    return f"❌ Lead {lead_id} not found"
            elif hasattr(odoo, 'execute_kw'):
                result = odoo.execute_kw(
                    ODOO_DB, uid, ODOO_PASSWORD,
                    'crm.lead', 'search_read',
                    [[('id', '=', int(lead_id))]],
                    {'fields': ['id', 'name', 'stage_id', 'partner_name', 'expected_revenue'], 'limit': 1}
                )
                if result:
                    lead = result[0]
                    return f"✅ Lead {lead_id} Status: {lead.get('stage_id', ['Unknown'])[1] if lead.get('stage_id') else 'Unknown'}, Partner: {lead.get('partner_name', 'Not Set')}, Revenue: ${lead.get('expected_revenue', 0):,.2f}"
                else:
                    return f"❌ Lead {lead_id} not found"
            else:
                return "❌ Odoo connection not available"
        else:
            return "❌ Please specify a lead number"
    except Exception as e:
        return f"❌ Error getting status: {str(e)}"

import re

def extract_values_by_model(message, model):
    """Extract values based on detected model"""
    print(f"🔍 Extracting values for model: {model}")
    print(f"📝 Message: {message}")
    
    values = {}
    
    # Extract lead number/ID
    lead_match = re.search(r'lead\s+no\s+(\d+)', message, re.IGNORECASE)
    if lead_match:
        values['lead_id'] = lead_match.group(1)
        print(f"🔢 Found lead ID: {values['lead_id']}")
    
    # Extract product name
    product_match = re.search(r'product\s+([A-Za-z0-9\s]+?)(?:\s+with|\s+qty|\s+and|$)', message, re.IGNORECASE)
    if product_match:
        values['product_name'] = product_match.group(1).strip()
        print(f"📦 Found product: {values['product_name']}")
    
    # Extract quantity
    qty_match = re.search(r'(\d+)\s+qty', message, re.IGNORECASE)
    if qty_match:
        values['quantity'] = int(qty_match.group(1))
        print(f"🔢 Found quantity: {values['quantity']}")
    
    # Extract customer name
    customer_patterns = [
        r'send\s+it\s+to\s+([A-Za-z\s]+)',
        r'to\s+([A-Za-z]+)(?:\s|$)',
        r'for\s+([A-Za-z\s]+)',
        r'customer\s+([A-Za-z\s]+)'
    ]
    
    for pattern in customer_patterns:
        customer_match = re.search(pattern, message, re.IGNORECASE)
        if customer_match:
            values['customer_name'] = customer_match.group(1).strip()
            print(f"👤 Found customer: {values['customer_name']}")
            break
    
    print(f"✅ Extracted values: {values}")
    return values
def process_message(message):
    """Main processing function"""
    print(f"🔧 Processing message: {message}")

    # Detect model early
    model = detect_model(message)
    if not model:
        return "❌ Could not determine what you want to do"
    print(f"🎯 Detected model: {model}")

    # Handle quotation creation first
    if 'quotation' in message.lower() or 'quote' in message.lower():
        values = extract_quotation_values(message)
        print(f"📝 Extracted quotation values: {values}")
        return create_quotation_from_lead(values)

    # Default flow for other operations
    intent = classify_intent(message)
    print(f"🧠 Detected intent: {intent}")

    record_id = resolve_record_id(message, model)
    values = parse_field_values(message, model)
    filters = extract_filters(message, model)
    fields = extract_fields(message, model)
    limit = extract_limit(message)

    print(f"🧠 Final Resolved Model: {model}")
    print(f"🧠 Final Resolved Record ID: {record_id}")
    print(f"🧠 Final Values: {values}")

    return execute_odoo_operation(
        intent=intent,
        model=model,
        filters=filters,
        fields=fields,
        values=values,
        record_id=record_id,
        limit=limit
    )


def send_quotation_email(order_id):
    """Send quotation by email using Odoo email template"""
    try:
        # Fetch email template for sale orders
        template_id = odoo.execute_kw(
            ODOO_DB, uid, ODOO_PASSWORD,
            'ir.model.data', 'xmlid_to_res_id',
            ['sale.email_template_edi_sale']
        )

        # Send email using the template
        result = odoo.execute_kw(
            ODOO_DB, uid, ODOO_PASSWORD,
            'mail.template', 'send_mail',
            [template_id, order_id],
            {'force_send': True}
        )

        return f"📧 Quotation email sent successfully (Mail ID: {result})"

    except Exception as e:
        logging.error(f"Error sending email: {e}")
        return "❌ Failed to send quotation email"
def download_quotation_pdf(order_id):
    """Download Quotation PDF for a sale.order using report service"""
    try:
        # Report name for quotation in Odoo
        report_name = 'sale.report_saleorder'
        pdf_content = odoo.execute_kw(
            ODOO_DB, uid, ODOO_PASSWORD,
            'report', 'get_pdf',
            [order_id, report_name]
        )
        filename = f"quotation_{order_id}.pdf"
        filepath = os.path.join('static', filename)
        with open(filepath, "wb") as f:
            f.write(pdf_content)

        return f"/static/{filename}"  # return URL path

    except Exception as e:
        logging.error(f"Error downloading PDF: {e}")
        return None
def create_quotation_from_lead(values):
    """Create quotation from existing lead using XML-RPC"""
    try:
        print(f"🔧 Creating quotation with values: {values}")
        
        lead_id = values.get('lead_id')
        if not lead_id:
            return "❌ Lead ID is required for quotation"

        # Step 1: Fetch lead to get partner_id
        lead_data = odoo.execute_kw(
            ODOO_DB, uid, ODOO_PASSWORD,
            'crm.lead', 'read',
            [[int(lead_id)]],
            {'fields': ['partner_id', 'id']}
        )
        if not lead_data:
            return f"❌ Lead ID {lead_id} not found"

        partner_id = lead_data[0].get('partner_id', False)
        if not partner_id:
            return f"❌ Lead {lead_id} has no customer (partner_id)"

        # Step 2: Prepare order_line if product is given
        order_lines = []
        if 'product_name' in values:
            product_name = values['product_name']
            quantity = values.get('quantity', 1)

            products = odoo.execute_kw(
                ODOO_DB, uid, ODOO_PASSWORD,
                'product.product', 'search_read',
                [[('name', 'ilike', product_name)]],
                {'fields': ['id', 'name', 'list_price'], 'limit': 1}
            )
            if not products:
                return f"❌ Product '{product_name}' not found"

            product = products[0]
            order_lines.append((0, 0, {
                'product_id': product['id'],
                'product_uom_qty': quantity,
                'name': product['name'],
                'price_unit': product.get('list_price', 0)
            }))

        # Step 3: Create the quotation (sale.order)
        quotation_data = {
            'partner_id': partner_id[0],
            'opportunity_id': int(lead_id),
            'order_line': order_lines
        }

        quotation_id = odoo.execute_kw(
            ODOO_DB, uid, ODOO_PASSWORD,
            'sale.order', 'create',
            [quotation_data]
        )

        return f"✅ Quotation created with ID {quotation_id} for Lead {lead_id}"

    except Exception as e:
        logging.error(f"Error in create_quotation_from_lead: {str(e)}")
        return f"❌ Error creating quotation: {str(e)}"

def create_or_update_lead(values):
    """Create or update lead"""
    try:
        if not odoo:
            return "❌ Odoo connection not available"
        
        lead_id = values.get('lead_id')
        
        if lead_id:
            # Update existing lead
            lead = odoo.env['crm.lead'].browse(int(lead_id))
            if lead.exists():
                update_data = {}
                if values.get('customer_name'):
                    update_data['partner_name'] = values['customer_name']
                lead.write(update_data)
                return f"✅ Updated Lead {lead_id}"
            else:
                return f"❌ Lead {lead_id} not found"
        else:
            # Create new lead
            lead_data = {
                'name': values.get('customer_name', 'New Lead'),
                'partner_name': values.get('customer_name')
            }
            lead = odoo.env['crm.lead'].create(lead_data)
            return f"✅ Created Lead {lead.id}"
    except Exception as e:
        return f"❌ Error with lead: {str(e)}"
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')
app.secret_key = '2e8e74217e37498fbf2a6d0c89314a009d3266246debeb1dbfce1e1544e50324'
@app.route('/process', methods=['POST'])
def process_request():
    try:
        data = request.get_json
        original_message = data.get('message', '')
        user_message = translate_to_english(original_message).strip()
        print("🧪 Type of translated message:", type(user_message))
        print("🧪 Value of translated message:", user_message)
        # 1️⃣ Classification check
        classification = classify_command(user_message)
        odoo_keywords = ['lead', 'customer', 'order', 'invoice', 'update', 'create', 'show', 'mark', 'won', 'lost']
        has_odoo_keywords = any(keyword in user_message.lower() for keyword in odoo_keywords)

        if not classification['is_odoo'] and classification['confidence'] < 0.5 and not has_odoo_keywords:
            return jsonify({
                "type": "fallback",
                "data": "⚠️ This doesn't seem like an Odoo command. Please ask about leads, customers, invoices, etc."
            })

        if user_message.startswith('odoo '):
            user_message = user_message[5:].strip()

        # 2️⃣ NLP Parsing
        intent = detect_intent(user_message)
        model = detect_model(user_message) or session.get("last_model")
        if not model:
            return jsonify({
                "type": "error",
                "data": "<p>❌ Could not identify the model. Please specify something like leads, customers, orders, etc.</p>"
            })

        filters = extract_filters(user_message, model)
        fields = extract_fields(user_message, model)
        if 'id' not in fields:
            fields.insert(0, 'id')
        limit = extract_limit(user_message)
        
        # 3️⃣ Enhanced record ID resolution
        record_id = None
        if intent in ['update', 'delete']:
            record_id = resolve_record_id(user_message, model)
        
        values = parse_field_values(user_message, model) if intent in ['create', 'update'] else None

        # 4️⃣ Save context for future use
        session["last_model"] = model
        if record_id:
            session["last_record"] = record_id

        # 5️⃣ Clarification check (for phone/amount confusion)
        if intent in ['create', 'update'] and values:
            validation_issues = validate_field_types(values)
            if validation_issues:
                return jsonify({
                    "type": "clarification",
                    "data": "⚠️ Some field values might be confused.",
                    "issues": validation_issues
                })

        # 6️⃣ Final execution
        result = execute_odoo_operation(
            intent=intent,
            model=model,
            filters=filters,
            fields=fields,
            values=values,
            record_id=record_id,
            limit=limit
        )

        # 7️⃣ Smart Memory Tracking for CREATE operations
        if intent == 'create' and 'Successfully created' in str(result):
            # Extract the created ID from the success message
            id_match = re.search(r'ID:\s*(\d+)', str(result))
            if id_match:
                session["last_record"] = int(id_match.group(1))

        # 8️⃣ Debug logs
        logging.debug(f"🧠 Final Resolved Model: {model}")
        logging.debug(f"🧠 Final Resolved Record ID: {record_id}")
        logging.debug(f"🧠 Session Model: {session.get('last_model')}")
        logging.debug(f"🧠 Session Record: {session.get('last_record')}")

        return jsonify({
            "type": "success",
            "data": result,
            "meta": {
                "intent": intent,
                "model": model,
                "filters": filters,
                "record_id": record_id,
                "record_count": len(result) if isinstance(result, list) else None
            }
        })

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({
            "type": "error",
            "data": f"<div class='error'>❌ Error processing request: {str(e)}</div>"
        })
if __name__ == '__main__':
    print("🚀 Starting Odoo AI Assistant...")
    print(f"🔗 Odoo URL: {ODOO_URL}")
    print(f"🗄️  Database: {ODOO_DB}")
    print(f"👤 User: {ODOO_USERNAME}")
    print("🤖 AI Assistant ready!")
    print(translate_to_english("mere paas naukri hai"))

    app.run(debug=True, host='0.0.0.0', port=5555)
